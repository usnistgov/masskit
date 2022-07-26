rem A cmd.exe wrapper script because GitLab CI/CD defaults to powershell
@echo OFF

call %0\..\init_arrow_env.bat
call %0\..\test_massspec_conda.bat