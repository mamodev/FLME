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


try:
    while True:
        msg = input('Enter message: ')
        n = 1

        if msg[0] == '$':
            parts = msg.split(' ')
            if len(parts) < 2:
                continue

            msg = ' '.join(parts[1:])
            n = int(parts[0][1:])

        msg = msg + '\n'
        for i in range(n):
            s.sendall(msg.encode())


except KeyboardInterrupt:
    print('KeyboardInterrupt')
    s.close()
    exit()


s.close()
