# clean all code via yapf
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { yapf --style='{based_on_style: pep8, column_limit: 99}' -i $_.FullName }
# clean all code via autopep8
# autopep8 --in-place --aggressive --aggressive -r .
flake8 .