@echo off
echo 🚀 CLUSTER TRANSFER SCRIPT
echo ========================

echo.
echo 🧪 Testing local code first...
python test_all_modules.py
if %errorlevel% neq 0 (
    echo ❌ Local tests failed!
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        echo Transfer cancelled.
        pause
        exit /b 1
    )
) else (
    echo ✅ Local tests passed!
)

echo.
echo 📤 Transferring to cluster...
echo.

echo 📁 Transferring negotiation_platform directory...
scp -r negotiation_platform s391129@julia2.hpc.uni-wuerzburg.de:/home/s391129/
if %errorlevel% neq 0 (
    echo ❌ Failed to transfer negotiation_platform
    pause
    exit /b 1
)

echo 📄 Transferring test files...
scp test_all_modules.py s391129@julia2.hpc.uni-wuerzburg.de:/home/s391129/
scp test_runner.py s391129@julia2.hpc.uni-wuerzburg.de:/home/s391129/

echo.
echo ✅ Transfer completed!
echo.
echo 🧪 Next steps:
echo 1. Connect: ssh s391129@julia2.hpc.uni-wuerzburg.de
echo 2. Test: python3 test_all_modules.py
echo 3. Run: python3 negotiation_platform/main.py
echo.
pause
