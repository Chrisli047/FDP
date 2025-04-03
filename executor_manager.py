from concurrent.futures import ProcessPoolExecutor

class ExecutorManager:
    """
    A singleton class to manage a global ProcessPoolExecutor.
    Ensures that all multiprocessing tasks share the same executor.
    """
    _executor = None  # Store a single executor instance

    @classmethod
    def get_executor(cls, max_workers=None):
        """
        Returns the global ProcessPoolExecutor instance.
        Initializes it if it doesn't exist.

        Parameters:
            max_workers (int, optional): Number of worker processes.
        """
        if cls._executor is None:
            cls._executor = ProcessPoolExecutor(max_workers=max_workers)  # ✅ Persistent pool
        return cls._executor

    @classmethod
    def shutdown(cls):
        """Shuts down the executor gracefully when the program ends."""
        if cls._executor is not None:
            cls._executor.shutdown(wait=True)
            cls._executor = None
