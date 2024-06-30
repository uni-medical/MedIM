# clean all code via yapf
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { yapf -i $_.FullName }