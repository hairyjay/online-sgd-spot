import boto3
import time

s3 = boto3.client('s3')

bucket = 'airline-stream'
key = 'airline.csv'
body = s3.get_object(Bucket=bucket, Key=key)['Body']

# number of bytes to read per chunk
chunk_size = 1000000

# the character that we'll split the data with (bytes, not string)
newline = '\n'
partial_chunk = ''

i = 0
while (True):
    chunk = partial_chunk + body.read(chunk_size).decode('latin-1')

    # If nothing was read there is nothing to process
    if chunk == '':
        break

    lines = chunk.split(newline)
    for j in range(len(lines) - 1):
        print(i, lines[j].split(','))
        i += 1
        time.sleep(0.05)

    #if i >= 25:
        #break
    # keep the partial line you've read here
    partial_chunk = lines[-1]
