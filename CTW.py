__author__ = 'Matthew Howe-Patterson'

import numpy as np
from itertools import chain, izip

class CTWNode:
    # Constructor
    def __init__(self, parentIndex, depth):
        self.LogProbability = 0  # Stores the log probability of the data partition this node has seen so far
        self.OneCount = 0  # Stores the number of 1's of the data partition this node has seen so far
        self.ZeroCount = 0  # Stores the number of 0's of the data partition this node has seen so far
        self.OneChildIndex = -1  # The index of the child node when the context splits on 1
        self.ZeroChildIndex = -1 # The index of the child node when the context splits on 0
        self.ParentNodeIndex = parentIndex  # The index of this nodes parent
        self.Depth = depth  # The depth in the tree of this node
        self.ChildRegister = []  # Child nodes register there log probabilities here during a recursive computation of
        # the log probability of the whole tree

    # Prints the instance variables of the node
    def printNode(self):
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
    def getDepth(self):
        return self.Depth

    # Called by child nodes during the recursive computation of log probability
    def registerLogProbability(self, logProbability):
        self.ChildRegister.append(logProbability)

    # Mutator for OneChildIndex property
    def registerOneChild(self, oneChildIndex):
        self.OneChildIndex = oneChildIndex

    # Mutator for ZeroChildIndex property
    def registerZeroChild(self, zeroChildIndex):
        self.ZeroChildIndex = zeroChildIndex

    # Updates the log probability of the node and increments the one count. Returns the index of the next node.
    def presentOneAndUpdate(self, cntxt):

        # Update the nodes log probability using KT estimator
        self.LogProbability = self.LogProbability + np.log(self.OneCount + 0.5) - np.log(
            self.OneCount + self.ZeroCount + 1)

        # Increment the nodes one count
        self.OneCount += 1

        # Return the index of the next child node given the context split
        contextBit = cntxt[self.Depth]

        if contextBit == 1:
            return self.OneChildIndex
        else:
            return self.ZeroChildIndex

    # Updates the log probability of the node and increments the zero count. Returns the index of the next node.
    def presentZeroAndUpdate(self, cntxt):

        # Update the nodes log probability using KT estimator
        self.LogProbability = self.LogProbability + np.log(self.ZeroCount + 0.5) - np.log(
            self.OneCount + self.ZeroCount + 1)

        # Increment the nodes zero count
        self.ZeroCount += 1

        # Return the index of the next child node given the context split
        contextBit = cntxt[self.Depth]

        if contextBit == 1:
            return self.OneChildIndex
        else:
            return self.ZeroChildIndex

    # Computes the log probability of the subtree under this node, after the children have registered there sub tree log probabilities.
    # Returns the value of the log probability and the index of this nodes parent.
    def computeLogProbability(self):

        # Average the probabilities of the child nodes and this node
        passValue = np.logaddexp(np.log(0.5) + self.LogProbability, np.log(0.5) + sum(self.ChildRegister))

        # Reset the log probability register
        self.ChildRegister = []

        # Return the log probability of this nodes subtree and this nodes parent index for registering
        return ([passValue, self.ParentNodeIndex])


class CTW:

    # Constructor
    def __init__(self, maxDepth):
        self.MaxDepth = maxDepth  # The context length that is used (maximum order of markov model)
        self.NodeList = [CTWNode(-1, 0)]  # Stores the trees nodes that have been instantiated

    # Prints the instance variables of all the nodes of the tree
    def printTree(self):
        print 'MaxDepth: ' + np.str(self.MaxDepth)
        for x in self.NodeList:
            x.printNode()

    # Recursively compute the log probability of the entire tree
    def computeLogProbability(self):

        # Starting with leaves, register each nodes log probability with it's parent
        for x in reversed(self.NodeList):

            [passValue, parentNodeIndex] = x.computeLogProbability()

            if parentNodeIndex >= 0:
                self.NodeList[parentNodeIndex].registerLogProbability(passValue)

        return passValue

    def presentBitStringAndUpdateWithSideInformation(self,bitString,sideInformation):

        for index in enumerate(bitString, start = self.MaxDepth):
            if index[0]+1 >= len(bitString):
                break

            cntxt = list(chain.from_iterable(izip(bitString[index[0]-self.MaxDepth:index[0]], sideInformation[(index[0]+1)-self.MaxDepth:(index[0]+1)])))

            if bitString[index[0]] == 1:
                self.presentOneAndUpdateInContext(cntxt)
            else:
                self.presentZeroAndUpdateInContext(cntxt)

    def presentBitStringAndUpdate(self,bitString):
        for index in enumerate(bitString, start = self.MaxDepth):
            if index[0]+1 >= len(bitString):
                break
            if bitString[index[0]+1] == 1:
                self.presentOneAndUpdateInContext(bitString[index[0]-self.MaxDepth:index[0]])
            else:
                self.presentZeroAndUpdateInContext(bitString[index[0]-self.MaxDepth:index[0]])

    def presentOneAndUpdateInContext(self,cntxt):
        tempcntxt = cntxt[::-1]
        notDone = 1
        index = 0
        count = 0
        while notDone == 1:

            if count >= self.MaxDepth:
                break

            nextIndex = self.NodeList[index].presentOneAndUpdate(tempcntxt)
            if nextIndex >= 0:
                index = nextIndex
            elif (nextIndex == -1) and (self.NodeList[index].getDepth() < self.MaxDepth):

                self.NodeList.append(CTWNode(index,self.NodeList[index].getDepth()+1))
                if tempcntxt[count] == 1:
                    self.NodeList[index].registerOneChild(len(self.NodeList)-1)
                else:
                    self.NodeList[index].registerZeroChild(len(self.NodeList)-1)
                index = self.NodeList.__len__() - 1
            else:
                notDone = 0
            count += 1

    def presentZeroAndUpdateInContext(self,cntxt):
        tempcntxt = cntxt[::-1]
        notDone = 1
        index = 0
        count = 0
        while notDone == 1:

            if count >= self.MaxDepth:
                break

            nextIndex = self.NodeList[index].presentZeroAndUpdate(tempcntxt)
            if nextIndex >= 0:
                index = nextIndex
            elif (nextIndex == -1) and (self.NodeList[index].getDepth() < self.MaxDepth):
                self.NodeList.append(CTWNode(index,self.NodeList[index].getDepth()+1))
                if tempcntxt[count] == 1:
                    self.NodeList[index].registerOneChild(len(self.NodeList)-1)
                else:
                    self.NodeList[index].registerZeroChild(len(self.NodeList)-1)
                index = self.NodeList.__len__() - 1
            else:
                notDone = 0
            count += 1