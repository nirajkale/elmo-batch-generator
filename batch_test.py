batch_size = 5
m = 12

data = list(range(m))

for index in range(20):
    batch_start = (index * batch_size)% m
    batch_end = ((index + 1) * batch_size) %m
    if batch_end <= batch_start:
        batch_end = m
    b = data[ batch_start: batch_end]
    print('I:',index,' | ', batch_start, ':', batch_end, ' | ', len(b))
