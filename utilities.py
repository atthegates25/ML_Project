class smart_dict(dict):
    def __missing__(self, key):
        return(0)

class LabelCountEncoder(object):
    def __init__(self):
        self.count_dict = smart_dict()

    def fit(self, column, rank_asc=True, rank_start_1=True):
        # This gives you a dictionary with level as the key and counts as the value
        # rank_asc determines whether values with higher counts are assigned lower encodings (rank_asc==True means the value with the highest count will be assigned encoding = 1)
        # rank_start_1 determines whether ranking will start at 1 (default)

        # create local variable for use in indexing sorted counts
        if rank_start_1 == True:
            rank_start = 1
        else:
            rank_start = 0

        # returns the values and counts sorted by counts
        count = sorted(column.value_counts().to_dict().items(), key=lambda x: x[1], reverse=rank_asc)
        # create index for zipping
        index = range(rank_start,len(count)+rank_start)
        # create dictionary of ranks
        self.count_dict = smart_dict(list(zip(map(lambda i: count[i-rank_start][0], index), index)))


    def __transform(self, column):
        # If a category only appears in the test set, we will assign the value to zero.
        # missing = 0 - assigned by smart_dict object
        # for now DO NOT use on its own - will return the None object if dict is empty (not sure why, need to fix)
        # use fix_transform instead
        if self.count_dict!= {}:
            column = column.map(self.count_dict)

        return(column)


    def fit_transform(self, column):
        self.fit(column)
        return self.__transform(column)
