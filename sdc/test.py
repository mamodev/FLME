# Create a socket and connect to TCP 
import socket
import os

# get port from argument

args = os.sys.argv
if len(args) > 1:
    port = int(args[1])



s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', port))

print('Connected to', s.getpeername())


def randBytes(n):
    return os.urandom(n)

while True:
    msg = randBytes(1024)
    s.sendall(msg)



s.close()
