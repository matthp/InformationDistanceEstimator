import numpy as np
from itertools import chain, izip


class CTWNode:
    # Constructor : Instances of this class are single nodes in the CTW tree, with pointers to parent node and child
    # nodes.
    def __init__(self, parent_index, depth):
        self.LogProbability = 0  # Stores the log probability of the data partition this node has seen so far
        self.OneCount = 0  # Stores the number of 1's of the data partition this node has seen so far
        self.ZeroCount = 0  # Stores the number of 0's of the data partition this node has seen so far
        self.OneChildIndex = -1  # The index of the child node when the context splits on 1
        self.ZeroChildIndex = -1 # The index of the child node when the context splits on 0
        self.ParentNodeIndex = parent_index  # The index of this nodes parent
        self.Depth = depth  # The depth in the tree of this node
        self.ChildRegister = []  # Child nodes register there log probabilities here during a recursive computation of
        # the log probability of the whole tree

    # Prints the instance variables of the node, used for debugging
    def print_node(self):
        print '\nNew Node:'
        print 'LogProbability: ' + np.str(self.LogProbability)
        print 'OneCount: ' + np.str(self.OneCount)
        print 'ZeroCount: ' + np.str(self.ZeroCount)
        print 'OneChildIndex: ' + np.str(self.OneChildIndex)
        print 'ZeroChildIndex: ' + np.str(self.ZeroChildIndex)
        print 'ParentNodeIndex: ' + np.str(self.ParentNodeIndex)
        print 'Depth: ' + np.str(self.Depth)
        print 'ChildRegister: ' + np.str(self.ChildRegister)

    # Accessor for Depth property
    def get_depth(self):
        return self.Depth

    # Called by child nodes during the recursive computation of log probability
    def register_log_probability(self, log_probability):
        self.ChildRegister.append(log_probability)

    # Mutator for OneChildIndex property
    def register_one_child(self, one_child_index):
        self.OneChildIndex = one_child_index

    # Mutator for ZeroChildIndex property
    def register_zero_child(self, zero_child_index):
        self.ZeroChildIndex = zero_child_index

    # Updates the log probability of the node and increments the one count. Returns the index of the next node.
    def present_one_and_update(self, cntxt):

        # Update the nodes log probability using KT estimator
        self.LogProbability = self.LogProbability + np.log(self.OneCount + 0.5) - np.log(
            self.OneCount + self.ZeroCount + 1)

        # Increment the nodes one count
        self.OneCount += 1

        # Return the index of the next child node given the context split
        context_bit = cntxt[self.Depth]

        if context_bit == 1:
            return self.OneChildIndex
        else:
            return self.ZeroChildIndex

    # Updates the log probability of the node and increments the zero count. Returns the index of the next node.
    def present_zero_and_update(self, cntxt):

        # Update the nodes log probability using KT estimator
        self.LogProbability = self.LogProbability + np.log(self.ZeroCount + 0.5) - np.log(
            self.OneCount + self.ZeroCount + 1)

        # Increment the nodes zero count
        self.ZeroCount += 1

        # Return the index of the next child node given the context split
        context_bit = cntxt[self.Depth]

        if context_bit == 1:
            return self.OneChildIndex
        else:
            return self.ZeroChildIndex

    # Computes the log probability of the subtree under this node, after the children have registered there sub tree
    # log probabilities.
    # Returns the value of the log probability and the index of this nodes parent.
    def compute_log_probability(self):

        # Average the probabilities of the child nodes and this node
        pass_value = np.logaddexp(np.log(0.5) + self.LogProbability, np.log(0.5) + sum(self.ChildRegister))

        # Reset the log probability register
        self.ChildRegister = []

        # Return the log probability of this nodes subtree and this nodes parent index for registering
        return ([pass_value, self.ParentNodeIndex])


class CTW:

    # Constructor : This class implements the context tree weighting algorithm. There are methods to update the model
    # with strings with or without side information, and get the log probability of the whole model.
    def __init__(self, max_depth):
        self.MaxDepth = max_depth  # The context length that is used (maximum order of markov model)
        self.NodeList = [CTWNode(-1, 0)]  # Stores the trees nodes that have been instantiated

    # Prints the instance variables of all the nodes of the tree
    def print_tree(self):
        print 'MaxDepth: ' + np.str(self.MaxDepth)
        for x in self.NodeList:
            x.print_node()

    # Recursively compute the log probability of the entire tree
    def compute_log_probability(self):

        # Starting with leaves, register each nodes log probability with it's parent
        for x in reversed(self.NodeList):

            [pass_value, parent_node_index] = x.compute_log_probability()

            if parent_node_index >= 0:
                self.NodeList[parent_node_index].register_log_probability(pass_value)

        return pass_value

    # Update the model with a string to model, and a side information string of the same length
    def present_bit_string_and_update_with_side_information(self, bit_string, side_information):

        for index in enumerate(bit_string, start=self.MaxDepth):
            if index[0]+1 >= len(bit_string):
                break

            cntxt = list(chain.from_iterable(izip(bit_string[index[0]-self.MaxDepth:index[0]], side_information[(index[0]+1)-self.MaxDepth:(index[0]+1)])))

            if bit_string[index[0]] == 1:
                self.present_one_and_update_in_context(cntxt)
            else:
                self.present_zero_and_update_in_context(cntxt)

    # Update the model with a string to model
    def present_bit_string_and_update(self, bit_string):
        for index in enumerate(bit_string, start=self.MaxDepth):
            if index[0]+1 >= len(bit_string):
                break
            if bit_string[index[0]+1] == 1:
                self.present_one_and_update_in_context(bit_string[index[0]-self.MaxDepth:index[0]])
            else:
                self.present_zero_and_update_in_context(bit_string[index[0]-self.MaxDepth:index[0]])

    # Called if current target bit is a one. 'cntxt' is the bit string context of the target.
    # Recursively updates / dynamically adds nodes to the tree with this information.
    def present_one_and_update_in_context(self, cntxt):
        temp_cntxt = cntxt[::-1]
        not_done = 1
        index = 0
        count = 0
        while not_done == 1:

            if count >= self.MaxDepth:
                break

            next_index = self.NodeList[index].present_one_and_update(temp_cntxt)
            if next_index >= 0:
                index = next_index
            elif (next_index == -1) and (self.NodeList[index].get_depth() < self.MaxDepth):

                self.NodeList.append(CTWNode(index, self.NodeList[index].get_depth()+1))
                if temp_cntxt[count] == 1:
                    self.NodeList[index].register_one_child(len(self.NodeList)-1)
                else:
                    self.NodeList[index].register_zero_child(len(self.NodeList)-1)
                index = self.NodeList.__len__() - 1
            else:
                not_done = 0
            count += 1

    # Called if current target bit is a zero. 'cntxt' is the bit string context of the target.
    # Recursively updates / dynamically adds nodes to the tree with this information.
    def present_zero_and_update_in_context(self, cntxt):
        temp_cntxt = cntxt[::-1]
        not_done = 1
        index = 0
        count = 0
        while not_done == 1:

            if count >= self.MaxDepth:
                break

            next_index = self.NodeList[index].present_zero_and_update(temp_cntxt)
            if next_index >= 0:
                index = next_index
            elif (next_index == -1) and (self.NodeList[index].get_depth() < self.MaxDepth):
                self.NodeList.append(CTWNode(index,self.NodeList[index].get_depth()+1))
                if temp_cntxt[count] == 1:
                    self.NodeList[index].register_one_child(len(self.NodeList)-1)
                else:
                    self.NodeList[index].register_zero_child(len(self.NodeList)-1)
                index = self.NodeList.__len__() - 1
            else:
                not_done = 0
            count += 1
