import requests
import time

def query_state(counter=0):
    post_data = {'type': 'query_result', 'idcard': '', 'phone': ''}
    post_data['idcard'] = '0037'
    post_data['phone'] = '18800267158'
    
    try:
        r = requests.post('http://join.qq.com/query/result', data=post_data)
        r.encoding = 'gb2313'
        data = r.text
        arr = data.split('</p>')
        
        # Check to ensure data split is successful
        if len(arr) > 1 and '<p>' in arr[1]:
            print(str(counter) + ':' + arr[1].split('<p>')[1])
        else:
            print(f"Error parsing data at counter {counter}: {data}")
        
    except Exception as e:
        print(f"Error at counter {counter}: {e}")

counter = 0
while True:
    query_state(counter)
    counter += 1
    time.sleep(10)
