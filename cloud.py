import time
import requests
import json
import datetime 

class cloud:
    def __init__(self):
        # project url
        self.firebase_url = 'https://m202a-final-project-default-rtdb.firebaseio.com/'  

    # 
    def setData(self, mask, temp):
        date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y/%m/%d %H:%M:%S')
        mask = str(mask)
        temp = str(temp)
        data = {'Date': date, 'Mask': mask, 'TEMP': temp}
        return data

    def service(self, mask, temp):
        data = self.setData(mask,temp)
        result = requests.post(self.firebase_url + '/Detection.json', data=json.dumps(data))
        print(data)
        print('Status Code = ' + str(result.status_code) + ', Response = ' + result.text)

# testing
if __name__ == "__main__":
    test = cloud()
    test.service("Mask", 36.5)