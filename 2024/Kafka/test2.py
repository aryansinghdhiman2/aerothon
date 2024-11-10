from  threading import Timer

def myfunc():
    print("hello")
    pass

def restartTimer(old_timer : Timer) -> Timer:
    old_timer.cancel()
    new_timer = Timer(interval=old_timer.interval,function=old_timer.function)
    new_timer.start()
    return new_timer

t = Timer(interval=5,function=myfunc)
t.start()
t.cancel()

t = restartTimer(t)
t.join()
# t.cancel()
# t.start()