class EliotWave:
    def __init__(self, x1, x2, x3, x4, x5, x6):
        self.x1 = x1.x6 if isinstance(x1, EliotWave) else x1
        self.x2 = x2.x6 if isinstance(x2, EliotWave) else x2
        self.x3 = x3.x6 if isinstance(x3, EliotWave) else x3
        self.x4 = x4.x6 if isinstance(x4, EliotWave) else x4
        self.x5 = x5.x6 if isinstance(x5, EliotWave) else x5
        self.x6 = x6.x6 if isinstance(x6, EliotWave) else x6

        if not self.is_valid():
            raise ValueError("Provided values do not satisfy the conditions for an Eliot Wave")

    def is_valid(self):
        condition1 = self.x1 < self.x3
        condition2 = self.x2 < self.x5
        condition3 = ((self.x2 - self.x1) < (self.x4 - self.x3)) or ((self.x6 - self.x5) < (self.x4 - self.x3))
        
        return condition1 and condition2 and condition3

'''
small_wave1 = EliotWave(1, 2, 3, 4, 5, 6)
small_wave2 = EliotWave(2, 3, 4, 5, 6, 7)
big_wave = EliotWave(small_wave1, small_wave2, 5, 6, 7, 8)
'''

