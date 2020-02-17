import requests

data = {'netid': 'cdf2', 'name': 'Christian Fernandez'}
url = "https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_string.php"

r = requests.post(url, data=data)
print(r.text[:400*498:498])