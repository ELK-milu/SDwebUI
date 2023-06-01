import torch
import os
import time
import multiprocessing
import signal

a=2

def stop():
    os.kill(mp, signal.SIGINT)
    return 0

def generation():
    global mp,a
    mp = multiprocessing.current_process().pid
    try:
        while True:
            a=a+1
            print(a)
            time.sleep(1)
            if a==5 :
                stop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping function")

    return a
    # return int(prompt)+int(steps)+int(seed)


if __name__ == "__main__":
    generation()