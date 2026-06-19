@echo off
cd /d "%~dp0"
if not exist "generated\downstream_88_evaluation" mkdir "generated\downstream_88_evaluation"
if exist "generated\downstream_88_evaluation\run_exit_code.txt" del "generated\downstream_88_evaluation\run_exit_code.txt"
echo start %DATE% %TIME% > "generated\downstream_88_evaluation\wrapper_trace.log"
echo cwd %CD% >> "generated\downstream_88_evaluation\wrapper_trace.log"
if exist "C:\Users\buckl\anaconda3\envs\env_py_3_12\python.exe" (
    echo python exists >> "generated\downstream_88_evaluation\wrapper_trace.log"
) else (
    echo python missing >> "generated\downstream_88_evaluation\wrapper_trace.log"
)
"C:\Users\buckl\anaconda3\envs\env_py_3_12\python.exe" -u evaluate_downstream_88.py > "generated\downstream_88_evaluation\run_stdout.log" 2> "generated\downstream_88_evaluation\run_stderr.log"
echo python finished %DATE% %TIME% code %ERRORLEVEL% >> "generated\downstream_88_evaluation\wrapper_trace.log"
echo %ERRORLEVEL% > "generated\downstream_88_evaluation\run_exit_code.txt"
