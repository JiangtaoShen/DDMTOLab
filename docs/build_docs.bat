@echo off
echo Cleaning old documentation...
rmdir /s /q _static _sources .doctrees 2>nul
del /q *.html *.js objects.inv .buildinfo searchindex.js 2>nul

echo Building documentation...
sphinx-build -b html source .

echo Done! Opening documentation...
start index.html