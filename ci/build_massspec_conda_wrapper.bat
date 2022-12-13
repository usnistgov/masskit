rem A cmd.exe wrapper script because GitLab CI/CD defaults to powershell
@echo OFF

rem set script_dir=%~dp0
rem call %script_dir%init_arrow_env.bat /f
rem call %script_dir%build_masskit_conda.bat
call ci\init_arrow_env.bat /f
call ci\build_masskit_conda.bat
