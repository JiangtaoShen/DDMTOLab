@echo off
REM release.bat - DDMTOLab Release Pipeline for Windows

echo =========================================
echo DDMTOLab Release Pipeline
echo =========================================

REM Configuration
set VERSION=1.0.0
set PACKAGE_NAME=ddmtolab

REM ========================================
REM Part 1: Cleanup
REM ========================================
echo.
echo Step 1: Cleaning old build files...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%i in (*.egg-info) do @rmdir /s /q "%%i"

REM ========================================
REM Part 2: PyPI Release
REM ========================================
echo.
echo Step 2: Building Python package...
python -m build

echo.
echo Step 3: Checking package integrity...
twine check dist/*

echo.
set /p test_pypi="Upload to TestPyPI for testing? (y/n): "
if /i "%test_pypi%"=="y" (
    echo Uploading to TestPyPI...
    twine upload --repository testpypi dist/*
    
    echo.
    echo Test installation command:
    echo pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ %PACKAGE_NAME%
    echo.
    pause
)

echo.
set /p upload_pypi="Upload to official PyPI? (y/n): "
if /i "%upload_pypi%"=="y" (
    echo Uploading to PyPI...
    twine upload dist/*
    echo ✓ PyPI release completed!
) else (
    echo ⏭  Skipped PyPI release
)

REM ========================================
REM Part 3: Git Tag & GitHub Release
REM ========================================
echo.
echo Step 4: Creating Git tag...
set /p create_tag="Create and push Git tag? (y/n): "
if /i "%create_tag%"=="y" (
    git tag -a "v%VERSION%" -m "Release version %VERSION%"
    git push origin "v%VERSION%"
    echo ✓ Git tag created and pushed
    echo.
    echo Please create a GitHub Release at:
    echo https://github.com/JiangtaoShen/DDMTOLab/releases/new
) else (
    echo ⏭  Skipped Git tag creation
)

REM ========================================
REM Summary
REM ========================================
echo.
echo =========================================
echo 🎉 Release pipeline completed!
echo =========================================
echo.
echo Installation commands:
echo   pip install %PACKAGE_NAME%
echo.
pause
