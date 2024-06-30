# pytest 
# pytest .\tests\test_stunet.py
# pytest .\tests\test_stunet.py::TestSTUNet_base
# pytest .\tests\test_stunet.py::TestSTUNet_large
# pytest .\tests\test_stunet.py::TestSTUNet_huge

# clean all code via yapf
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { yapf -i $_.FullName }