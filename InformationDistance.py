__author__ = 'Matthew Howe-Patterson'

from CTW import CTW

def estimateInfoDistance(X,Y, contextLength):

    # Return -1 if the lengths of the strings do not match
    if len(X) != len(Y):
        return -1

    # Estimate the entropy of X using CTW
    Cx = CTW(contextLength)
    Cx.presentBitStringAndUpdate(X)
    Hx = -Cx.computeLogProbability() / len(X)
    Cx = []

    # Estimate the entropy of Y using CTW
    Cy = CTW(contextLength)
    Cy.presentBitStringAndUpdate(Y)
    Hy = -Cy.computeLogProbability() / len(Y)
    Cy = []

    # Estimate the entropy of X conditioned on Y using CTW
    Cxy = CTW(contextLength)
    Cxy.presentBitStringAndUpdateWithSideInformation(X,Y)
    Hxy = -Cxy.computeLogProbability() / len(X)
    Cxy = []

    # Estimate the entropy of Y conditioned on X using CTW
    Cyx = CTW(contextLength)
    Cyx.presentBitStringAndUpdateWithSideInformation(Y,X)
    Hyx = -Cyx.computeLogProbability() / len(Y)
    Cyx = []

    # Compute the estimated normalized information distance between X and Y
    return max([Hxy,Hyx]) / max([Hx, Hy])