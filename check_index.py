from app import app
with app.test_client() as c:
    r = c.get('/')
    print(r.status_code)
    print(r.data.decode('utf-8')[:300])
