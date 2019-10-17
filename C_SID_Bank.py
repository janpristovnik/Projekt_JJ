
class Banka(object):
    """SID Banka """
    def __init__(self, *args, **kwargs):
        self.total_ammount = 1.0

    def new_loans(self, iAmmount):
        self.total_ammount -= iAmmount
