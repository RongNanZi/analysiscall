class modelx(object):
    def __init__(self,
                embed,
                max_length,
                 x,
                 padding0=True
                ):
        self.data = x
        self.embedding_matrix  =  embed
        self.max_length = max_length
        self.padding0 = padding0
