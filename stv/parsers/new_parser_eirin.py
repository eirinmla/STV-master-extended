from enum import Enum
from .parser import Parser

class UpgradeFormula():
    upgrades = []
    expression = None
    agents = []

    def __init__(self):
        pass

    def __str__(self):
        return (", ".join(self.upgrades)) + "<<" + (", ".join(self.agents)) + ">>" + str(self.expression)

class UpgradeType(Enum):
    P = "+"
    N = "-"

class Upgrade():
    upgradeType = None
    updates = []

class Update():
    from_state = []
    agent = None
    to_states = []

    def evaluate(self, state):
        if "[" in state:
            pass # må kalle på alle parserene
        # skal returnere parsingen
        elif "<" in state: 
            pass # må kalle på to parsere
        else: 
            pass # må kalle på formulaparserexpression


class FormulaParser(Parser):

    def __init__(self):
        pass

    def parseUpgradeFormula(self, formulaStr):
        self.setStr(formulaStr)

        formula = UpgradeFormula()
        formula.upgrades = self.__parseFormulaUpgrades()
        print(formula.upgrades)
        formula.agents = self.__parseFormulaAgents()
        print(formula.agents)
        formula.expression = self.__parseFormulaExpression()
        print(formula.expression)

        return formula 


    def __parseFormulaAgents(self):
        agents = []
        self.consume("<<")
        while True:
            res = self.readUntil([">", ","])
            str = res[0]
            chr = res[1]
            agents.append(str)
            if chr == ">":
                break
            else:
                self.consume(",")
        self.consume(">>")
        return agents


def __parseFormulaExpression(self):
        self.consume("(")
        formulaExpression = []
        while True:
            res = self.readUntil([")", "(", "&", "|", "=", "!", ">"])
            str = res[0]
            chr = res[1]
            if chr == ")":
                if len(str) > 0:
                    formulaExpression.append(str)
                break
            elif chr == "(":
                formulaExpression.append(self.__parseFormulaExpression())
            elif chr == "&" or chr == "|" or chr == "=" or chr == "!" or chr == ">":
                if len(str) > 0:
                    formulaExpression.append(str)
                if chr == "!" and self.peekChar(1) == "=":
                    formulaExpression.append("!=")
                    self.stepForward()
                else:
                    formulaExpression.append(chr)
                self.stepForward()
            else:
                raise Exception("Unimplemented character inside __parseFormulaExpression")
        print("formulaExpression", formulaExpression)
        simpleExpression = self.__convertToSimpleExpression(formulaExpression)
        print("simpleExpression", simpleExpression)
        self.consume(")")
        return simpleExpression

    def __convertToSimpleExpression(self, arr):
        # Single value
        if not isinstance(arr, list):
            return arr
        if len(arr) == 1:
            return arr[0]

        # NOT
        i = 0
        l = len(arr)
        while i < l:
            if arr[i] == "!":
                if i + 1 < l:
                    arr.pop(i)
                    l = l - 1
                    arr[i] = SimpleExpression(None, SimpleExpressionOperator.NOT, arr[i])
            i = i + 1

        # OR
        if arr.count("|") > 0:
            return self.__convertToSimpleExpressionByOperator(arr, SimpleExpressionOperator.OR)

        # AND
        if arr.count("&") > 0:
            return self.__convertToSimpleExpressionByOperator(arr, SimpleExpressionOperator.AND)

        # EQ/NEQ
        if len(arr) == 3 and (arr[1] == "=" or arr[1] == "!=" or arr[1] == ">"):
            left = self.__convertToSimpleExpression(arr[0])
            right = self.__convertToSimpleExpression(arr[2])
            if arr[1] == "=":
                return SimpleExpression(left, SimpleExpressionOperator.EQ, right)
            elif arr[1] == "!=":
                return SimpleExpression(left, SimpleExpressionOperator.NEQ, right)
            elif arr[1] == ">":
                return SimpleExpression(left, SimpleExpressionOperator.GT, right)

        return arr

    def __convertToSimpleExpressionByOperator(self, arr, op):
        i = 0
        l = len(arr)
        parts = []
        part = []
        while i < l:
            if arr[i] == op.value:
                parts.append(part)
                part = []
            else:
                part.append(arr[i])
            i = i + 1
        parts.append(part)
        for i in range(len(parts)):
            parts[i] = self.__convertToSimpleExpression(parts[i])
        expr = SimpleExpression(parts[0], op, parts[1])
        innermostExpr = expr
        for i in range(2, len(parts)):
            innermostExpr.right = SimpleExpression(innermostExpr.right, op, parts[i])
            innermostExpr = innermostExpr.right
        return expr
