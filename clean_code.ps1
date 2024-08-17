# clean all code via yapf
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { yapf -i $_.FullName }
# clean all code via autopep8
# autopep8 --in-place --aggressive --aggressive -r .
flake8 .