import requests

response = requests.get('https://api.github.com/this-api-should-not-exist')
print(response.status_code)

response = requests.get('https://api.github.com')
print(response.status_code)

if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')

response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")

response=response.json()

response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
)

response=response.json()

response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')

with open(r'img.png','wb') as f:
    f.write(response.content)
    
pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)

response=response.json()
