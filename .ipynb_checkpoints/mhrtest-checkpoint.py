class thetest:
    def __init__(self):
        self.mydict = {'t': 1}

    def apply(self):
        self.mydict['t'] = 2

this = thetest()
mydicthere = dict(this.mydict)
this.apply()
print(mydicthere)