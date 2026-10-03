"""
Очередь задач с FIFO-дисциплиной и настраиваемой степенью параллелизма.

Все задачи проходят через одну очередь. Количество одновременно
выполняемых задач регулируется параметром max_parallel: 1 — строгая
последовательная обработка (FIFO), 2+ — параллельная с ограничением.

При изменении max_parallel на лету:
  - увеличение — сразу добавляются новые воркеры
  - уменьшение — «лишние» воркеры завершаются после текущей задачи

Задачи, ожидающие в очереди, находятся в статусе 'queued' и имеют
позицию (1-based). Позиция 1 — следующая на выполнение.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Dict, List, Optional

MAX_PARALLEL_HARD_LIMIT = 16


class TaskQueue:
    """
    Очередь задач с пулом воркеров.

    Воркеры запускаются под каждый «слот» параллелизма. Каждый воркер
    работает в цикле: берёт job_id из очереди, вызывает run_job, ждёт
    следующую задачу. Если worker_id >= текущего max_parallel (после
    уменьшения настройки), воркер завершается самостоятельно.

    Attributes:
        _queue: FIFO-очередь job_id.
        _workers: Потоки-воркеры.
        _max_parallel: Текущий лимит одновременных задач.
        _get_job: Callback получения Job по id.
        _run_job: Callback запуска транскрибации.
        _on_change: Callback для уведомления UI об изменении очереди.
        _positions: Кэш позиций (job_id -> 1-based позиция).
        _running_ids: Множество job_id, выполняемых прямо сейчас.
    """

    def __init__(
        self,
        get_job: Callable[[str], object],
        run_job: Callable[[object], None],
        on_change: Optional[Callable[[], None]] = None,
    ):
        """
        Создаёт очередь. Воркеры запускаются отдельно через start().

        Args:
            get_job: Функция, возвращающая Job по id (или None).
            run_job: Функция обработки задачи (запускает транскрибацию).
            on_change: Опциональный callback, вызывается при любом
                изменении очереди — для обновления UI.
        """
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._workers: List[threading.Thread] = []
        self._max_parallel = 1
        self._shutdown = threading.Event()
        self._get_job = get_job
        self._run_job = run_job
        self._on_change = on_change
        self._positions: Dict[str, int] = {}
        self._running_ids: set = set()

    def start(self, max_parallel: int = 1) -> None:
        """
        Запускает очередь с указанным уровнем параллелизма.

        Args:
            max_parallel: Начальное количество одновременных задач.
                Зажимается в [1, MAX_PARALLEL_HARD_LIMIT].
        """
        self._max_parallel = self._clamp(max_parallel)
        self._spawn(self._max_parallel)

    def _clamp(self, n: int) -> int:
        """
        Приводит значение к допустимому диапазону.

        Args:
            n: Произвольное целое.

        Returns:
            Значение в [1, MAX_PARALLEL_HARD_LIMIT].
        """
        return max(1, min(int(n), MAX_PARALLEL_HARD_LIMIT))

    def _spawn(self, target: int) -> None:
        """
        Создаёт недостающих воркеров.

        Уже живые воркеры не трогаются — добавляются только новые
        с индексами от текущего количества до target.

        Args:
            target: Целевое количество активных воркеров.
        """
        with self._lock:
            alive = [w for w in self._workers if w.is_alive()]
            self._workers = alive
            start_idx = len(alive)
            for i in range(start_idx, target):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(i,),
                    daemon=True,
                    name=f"whisper-worker-{i}",
                )
                t.start()
                self._workers.append(t)

    def _worker_loop(self, worker_id: int) -> None:
        """
        Основной цикл воркера.

        Проверяет, не «уволен» ли он, берёт задачи из очереди и
        передаёт их в run_job. Завершается по shutdown или когда его
        worker_id становится больше текущего max_parallel.

        Args:
            worker_id: Индекс воркера (0-based).
        """
        while not self._shutdown.is_set():
            with self._lock:
                if worker_id >= self._max_parallel:
                    return

            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if job_id is None:
                return

            with self._lock:
                if worker_id >= self._max_parallel:
                    self._queue.put(job_id)
                    return

            job = None
            try:
                job = self._get_job(job_id)
                if job is None:
                    continue
                if job.is_cancelled():
                    job.set_status("cancelled", "Отменено до старта")
                    job.finished_at = time.time()
                    job.persist_final()
                    job.emit({"type": "cancelled"})
                    continue

                with self._lock:
                    self._running_ids.add(job_id)
                    self._recompute_positions_locked()

                job.set_status("pending", "Задача запущена")
                self._run_job(job)
            except Exception as e:
                print(f"[queue] worker {worker_id} error: {e}")
            finally:
                with self._lock:
                    self._running_ids.discard(job_id)
                    self._positions.pop(job_id, None)
                    self._recompute_positions_locked()
                self._notify()

    def _recompute_positions_locked(self) -> None:
        """
        Пересчитывает позиции задач в очереди.

        Вызывается под self._lock. Перебирает очередь, очищает кэш и
        восстанавливает элементы в том же порядке.
        """
        self._positions.clear()
        items: List[str] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            items.append(item)

        for i, item in enumerate(items):
            if item is not None:
                self._positions[item] = i + 1
            self._queue.put(item)

    def put(self, job_id: str) -> None:
        """
        Добавляет задачу в конец очереди.

        Args:
            job_id: Идентификатор задачи.
        """
        with self._lock:
            self._queue.put(job_id)
            self._recompute_positions_locked()
        self._notify()

    def remove(self, job_id: str) -> bool:
        """
        Удаляет задачу из очереди, если она там есть.

        Используется при отмене: если задача ещё не взята воркером,
        её нужно убрать из очереди явно. Если уже взята — вернёт False,
        и вызывающая сторона должна полагаться на флаг is_cancelled.

        Args:
            job_id: Идентификатор задачи.

        Returns:
            True, если задача была найдена и удалена из очереди.
        """
        with self._lock:
            items: List[str] = []
            removed = False
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item == job_id:
                    removed = True
                else:
                    items.append(item)

            for item in items:
                self._queue.put(item)

            if removed:
                self._positions.pop(job_id, None)
                self._recompute_positions_locked()

        if removed:
            self._notify()
        return removed

    def position(self, job_id: str) -> Optional[int]:
        """
        Возвращает 1-based позицию задачи в очереди.

        Args:
            job_id: Идентификатор задачи.

        Returns:
            Позиция или None, если задача уже не в очереди.
        """
        with self._lock:
            return self._positions.get(job_id)

    def positions(self) -> Dict[str, int]:
        """
        Возвращает копию словаря позиций.

        Returns:
            Словарь {job_id: позиция} для всех задач в очереди.
        """
        with self._lock:
            return dict(self._positions)

    def size(self) -> int:
        """
        Возвращает количество задач, ожидающих в очереди.

        Returns:
            Количество job_id в очереди.
        """
        return self._queue.qsize()

    def running_count(self) -> int:
        """
        Возвращает количество задач, выполняемых прямо сейчас.

        Returns:
            Число активных задач.
        """
        with self._lock:
            return len(self._running_ids)

    def set_max_parallel(self, n: int) -> int:
        """
        Изменяет лимит одновременных задач на лету.

        При увеличении сразу создаются новые воркеры. При уменьшении
        «лишние» воркеры завершатся сами: они проверяют свой worker_id
        перед взятием следующей задачи.

        Args:
            n: Новый лимит (зажимается в допустимый диапазон).

        Returns:
            Актуальное значение после зажатия.
        """
        n = self._clamp(n)
        with self._lock:
            self._max_parallel = n
        alive_count = len([w for w in self._workers if w.is_alive()])
        if n > alive_count:
            self._spawn(n)
        self._notify()
        return n

    def get_max_parallel(self) -> int:
        """
        Возвращает текущий лимит одновременных задач.

        Returns:
            Значение max_parallel.
        """
        with self._lock:
            return self._max_parallel

    def _notify(self) -> None:
        """Вызывает on_change, если он задан."""
        if self._on_change:
            try:
                self._on_change()
            except Exception:
                pass

    def shutdown(self, timeout: float = 3.0) -> None:
        """
        Останавливает очередь и все воркеры.

        Активные задачи не прерываются — воркеры завершатся после того,
        как закончат текущую. Задачи в очереди останутся со статусом
        queued и будут помечены как interrupted при следующем старте.

        Args:
            timeout: Максимальное время ожидания завершения воркеров.
        """
        self._shutdown.set()
        for _ in self._workers:
            try:
                self._queue.put(None, timeout=0.1)
            except queue.Full:
                pass
        for w in self._workers:
            w.join(timeout=timeout / max(len(self._workers), 1))


_queue_singleton: Optional[TaskQueue] = None


def init_task_queue(
    get_job: Callable[[str], object],
    run_job: Callable[[object], None],
    max_parallel: int = 1,
    on_change: Optional[Callable[[], None]] = None,
) -> TaskQueue:
    """
    Инициализирует глобальную очередь задач.

    Args:
        get_job: Callback получения Job по id.
        run_job: Callback запуска обработки.
        max_parallel: Начальное количество одновременных задач.
        on_change: Callback уведомления об изменениях.

    Returns:
        Инициализированный TaskQueue.
    """
    global _queue_singleton
    _queue_singleton = TaskQueue(get_job, run_job, on_change=on_change)
    _queue_singleton.start(max_parallel)
    return _queue_singleton


def get_task_queue() -> TaskQueue:
    """
    Возвращает глобальную очередь задач.

    Returns:
        Singleton TaskQueue.

    Raises:
        RuntimeError: Если init_task_queue() не был вызван.
    """
    if _queue_singleton is None:
        raise RuntimeError("TaskQueue не инициализирована. Вызовите init_task_queue().")
    return _queue_singleton
