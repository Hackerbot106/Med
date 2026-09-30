import requests
from bs4 import BeautifulSoup

s = requests.Session()
# 1. Fetch CSRF token
r = s.get('http://127.0.0.1:5000/login')
soup = BeautifulSoup(r.text, 'html.parser')
csrf = soup.find('input', {'name': 'csrf_token'})['value']
print(f'CSRF Token: {csrf}')

# Test invalid normal login
r = s.post('http://127.0.0.1:5000/login', data={'csrf_token': csrf, 'username': 'worker1', 'password': 'wrongpassword'}, allow_redirects=False)
print(f'Invalid Login Status: {r.status_code}, URL: {r.headers.get("Location")}')

# Test normal login
r = s.post('http://127.0.0.1:5000/login', data={'csrf_token': csrf, 'username': 'worker1', 'password': 'demo123'}, allow_redirects=False)
print(f'Valid Login Status: {r.status_code}, Redirect: {r.headers.get("Location")}')

s.get('http://127.0.0.1:5000/logout')

# Test demo worker
r = s.get('http://127.0.0.1:5000/login')
soup = BeautifulSoup(r.text, 'html.parser')
csrf = soup.find('input', {'name': 'csrf_token'})['value']
r = s.post('http://127.0.0.1:5000/login', data={'csrf_token': csrf, 'demo_role': 'worker'}, allow_redirects=False)
print(f'Worker Demo Login Status: {r.status_code}, Redirect: {r.headers.get("Location")}')
r2 = s.get('http://127.0.0.1:5000' + r.headers.get("Location"), allow_redirects=False)
print(f'Worker redirected to: {r2.headers.get("Location")}')

s.get('http://127.0.0.1:5000/logout')

# Test demo reviewer
r = s.get('http://127.0.0.1:5000/login')
soup = BeautifulSoup(r.text, 'html.parser')
csrf = soup.find('input', {'name': 'csrf_token'})['value']
r = s.post('http://127.0.0.1:5000/login', data={'csrf_token': csrf, 'demo_role': 'reviewer'}, allow_redirects=False)
print(f'Reviewer Demo Login Status: {r.status_code}, Redirect: {r.headers.get("Location")}')
r2 = s.get('http://127.0.0.1:5000' + r.headers.get("Location"), allow_redirects=False)
print(f'Reviewer redirected to: {r2.headers.get("Location")}')

s.get('http://127.0.0.1:5000/logout')

# Test demo auditor
r = s.get('http://127.0.0.1:5000/login')
soup = BeautifulSoup(r.text, 'html.parser')
csrf = soup.find('input', {'name': 'csrf_token'})['value']
r = s.post('http://127.0.0.1:5000/login', data={'csrf_token': csrf, 'demo_role': 'auditor'}, allow_redirects=False)
print(f'Auditor Demo Login Status: {r.status_code}, Redirect: {r.headers.get("Location")}')
r2 = s.get('http://127.0.0.1:5000' + r.headers.get("Location"), allow_redirects=False)
print(f'Auditor redirected to: {r2.headers.get("Location")}')

s.get('http://127.0.0.1:5000/logout')
