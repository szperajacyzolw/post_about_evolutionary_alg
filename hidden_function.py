'''Tekne Consulting blog post --- teknecons.com '''


def hidden_fun(x, y):
    if x < 0 and y < 0:
        z = 500 + (x + 5)**4 + (y + 5)**4
    elif x < 0 and y > 0:
        z = 250 + (x + 5)**4 + (y - 5)**4
    elif x > 0 and y < 0:
        z = 30 + (x - 5)**4 + (y + 5)**4
    else:
        z = 0 + (x - 5)**4 + (y - 5)**4
    return(z)
