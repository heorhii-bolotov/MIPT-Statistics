import vk
import pandas as pd
import time

session = vk.Session(access_token='995d51d3bf60cb945fd7cd00b3be3c4a7ca8fb502ffe6c540ab7ec43c09ef84516b5ce6724440f9b4f8ef')
api = vk.API(session)
api.users.get(user_ids=1)


def getGroupIds():
    response = api.groups.getMembers(group_id='diht2015')
    return response['users']


ids = getGroupIds()
sample = []


def makeResponse(responseMaker):
    if not hasattr(makeResponse, 'delay'):
        makeResponse.delay = 0

    while True:
        try:
            time.sleep(makeResponse.delay)
            return responseMaker()
        except vk.exceptions.VkAPIError as e:
            if e.code == 6:
                makeResponse.delay += 0.1
                print('Увеличили задержку до {:.1f} секунд'.format(makeResponse.delay))
                time.sleep(5)


for i, id in enumerate(ids):
    print(i + 1, '/', len(ids))


    def getUserCounters():
        return api.users.get(user_ids=id, fields='counters')


    response = makeResponse(getUserCounters)
    user = response[0]
    if 'counters' in user:
        number_friends = user['counters']['friends']
        sample.append(number_friends)

with open('/home/dima/stats/practice/1/friends.csv', 'w') as file:
    file.write('\n'.join(map(str, sample)))
