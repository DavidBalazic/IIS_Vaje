from pathlib import Path

output = Path("reports/index.html")
tests = Path("reports/tests")
validations = Path("reports/validations")

def list_links(folder):
    return [
        f'<li><a href="{folder.name}/{f.name}">{f.name}</a></li>'
        for f in folder.glob("*.html")
    ]

html = f"""
<!DOCTYPE html>
<html>
<head><title>Reports</title></head>
<body>
<h1>Validation Reports</h1>
<ul>{''.join(list_links(validations))}</ul>
<h1>Test Reports</h1>
<ul>{''.join(list_links(tests))}</ul>
</body>
</html>
"""

output.write_text(html)