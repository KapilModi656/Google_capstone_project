
def execute_code(code: str, exec_globals: dict|None = None) -> dict:
    """
    Executes a block of Python code, captures stdout, and returns a serializable
    result dict containing only the captured output and a list of top-level
    variable names. This avoids returning module or object references which
    cannot be pickled by the ADK runner.

    Args:
        code (str): The Python code to execute.
        exec_globals (dict, optional): A dictionary to use as the global namespace during execution.

    Returns:
        dict: A dictionary with keys:
            - 'stdout': the captured printed output (string)
            - 'result_keys': list of top-level names defined by the executed code
    """
    import io
    import contextlib

    if exec_globals is None:
        exec_globals = {}

    exec_locals = {}

    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, exec_globals, exec_locals)
    except Exception as e:
     
        import traceback

        tb = traceback.format_exc()
        return {
            "stdout": stdout_buf.getvalue() + "\n" + tb,
            "result_keys": list(exec_locals.keys()),
            "error": True,
            "error_message": str(e),
        }

    return {
        "stdout": stdout_buf.getvalue(),
        "result_keys": list(exec_locals.keys()),
    }