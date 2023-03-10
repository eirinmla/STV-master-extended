from enum import Enum
from .parser import Parser


class TemporalOperator(Enum):
    F = "F"
    G = "G"
    FG = "FG"
    GF = "GF"
    X = "X"


class PathQuantifier(Enum):
    A = "A"
    E = "E"


class UpgradeType(Enum):
    P = "+"
    N = "-"

class Formula:
    expression = None
    temporalOperator = None

    def __init__(self):
        pass

    def __str__(self):
        return str(self.temporalOperator.value) + str(self.expression)


class AtlFormula(Formula):
    agents = []

    def __str__(self):
        return "<<" + (", ".join(self.agents)) + ">>" + super().__str__()


class CtlFormula(Formula):
    pathQuantifier = None

    def __str__(self):
        return str(self.pathQuantifier.value) + super().__str__()

class UpgradeFormula():
    upgrades = []
    upgradeType = None
    expression = None
    agents = []

    def __init__(self):
        pass

    def __str__(self):
        return "[" + (", ".join(self.upgrades)) + "]" + self.upgradeType.value + "<<" + (", ".join(self.agents)) + ">>" + str(self.expression)

class UpgradeExpression():
    upgradeList = []
    coalitionExpression = None

    def __init__(self):
        pass

    def __str__(self):
        if upgradeList:
            return str(self.upgradeList) + str(self.coalitionExpression)
        else:
            return str(self.coalitionExpression)

class CoalitionExpression():
    coalition = []
    simpleExpression = None

    def __init__(self):
        pass

    def __str__(self):
        if coalition:
            return str(self.coalition) + str(self.simpleExpression)
        else:
            return str(self.simpleExpression)

class UpgradeList():
    upgrades = None

    def __init__(self):
        pass

    def __str__(self):
        return '{' + ', '.join(self.upgrades) + '}'

class Upgrade():
    updates = None

    def __init__(self):
        pass
    
    def __str__(self):
        return '[' + ', '.join(self.updates) + ']'

class Update():
    fromState = None
    agent = None
    toState = None
    upgradeType = None

    def __init__(self):
        pass
    
    def __str__(self):
        return '(' + str(self.fromState) + ',' + str(self.agent) + ',' + str(self.toState) + ')' + str(self.upgradeType.value)

class SimpleExpressionOperator(Enum):
    AND = "&"
    OR = "|"
    NOT = "!"
    EQ = "="
    NEQ = "!="
    GT = ">"


class SimpleExpression:
    left = None
    operator = None
    right = None

    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __getValue(self, item, varValues):
        if isinstance(item, str):
            if item in varValues:
                return varValues[item]
            else:
                return item
        elif isinstance(item, SimpleExpression):
            return item.evaluate(varValues)
        return item

    def evaluate(self, varValues):
        left = self.__getValue(self.left, varValues)
        right = self.__getValue(self.right, varValues)
        if self.operator == SimpleExpressionOperator.NOT:
            return not right
        elif self.operator == SimpleExpressionOperator.AND:
            return not not (left and right)
        elif self.operator == SimpleExpressionOperator.OR:
            return not not (left or right)
        elif self.operator == SimpleExpressionOperator.EQ:
            return str(left) == str(right)
        elif self.operator == SimpleExpressionOperator.NEQ:
            return str(left) != str(right)
        elif self.operator == SimpleExpressionOperator.GT:
            return int(left) > int(right)
        else:
            raise Exception("Can't evaluate a SimpleExpression: unknown operator")

    def __str__(self):
        if self.operator == SimpleExpressionOperator.NOT:
            return str(self.operator.value) + str(self.right)
        else:
            return "(" + str(self.left) + " " + str(self.operator.value) + " " + str(self.right) + ")"

class Agent():
    literal = None
    def __init__(self, atom):
        self.literal = atom

    def __str__(self):
        return self.literal


class FormulaParser(Parser):

    def __init__(self):
        pass

    def parseAtlFormula(self, formulaStr):
        self.setStr(formulaStr)

        formula = AtlFormula()
        formula.agents = self.__parseFormulaAgents()
        formula.temporalOperator = self.__parseFormulaTemporalOperator()
        formula.expression = self.__parseFormulaExpression()

        return formula

    def parseCtlFormula(self, formulaStr):
        self.setStr(formulaStr)

        formula = CtlFormula()
        formula.pathQuantifier = self.__parseFormulaPathQuantifier()
        formula.temporalOperator = self.__parseFormulaTemporalOperator()
        formula.expression = self.__parseFormulaExpression()

        return formula
    
    def parseUpgradeFormula(self, formulaStr):
        self.setStr(formulaStr)

        formula = UpgradeFormula()
        formula.upgrades = self.__parseFormulaUpgrades()
        print(formula.upgrades)
        formula.agents = self.__parseFormulaAgents()
        print(formula.agents)
        formula.upgradeType = self.__parseFormulaUpgradeType(formula.upgrades)
        print(formula.upgradeType)
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

    def __parseFormulaUpgrades(self):
        upgrades = []
        self.consume("{")
        while True:
            res = self.readUntil([",", "}"])
            str = res[0]
            chr = res[1]
            if "(" in str:
                if "[" in str:
                    upgrades.append(str[0])
                    upgrades.append(str[2:])
                else:  
                    upgrades.append(str[1:])
            elif ")" in str:
                if "]" in str: 
                    upgrades.append(str[:-3])
                    upgrades.append(str[-2])
                    upgrades.append(str[-1])
                else: 
                    upgrades.append(str[:-2])
                    upgrades.append(str[-1])
            else:
                upgrades.append(str)

            if chr == "}":
                break
            else:
                self.consume(",")
        self.consume("}")
        upgrades_list = []
        for element in upgrades:
            if element == "[":
                temp_list = []
            if element != "[" and element != "]" and element != "(" and element != ")":
                temp_list.append(element)
            if element == "]":
                upgrades_list.append(temp_list)
        upgrades = upgrades_list
        return upgrades
    
    def __parseFormulaUpgradeType(self, formula): #  notes if the upgrade is positive or negative, it takes the last sign of the formula which will always be + or -, 
                                                  #  and all updates in the same upgrade is of the same type. 
        upgrade_type_list = []
        for element in formula: 
            if (element[-1]) == "+":
                upgrade_type_list.append(UpgradeType.P)
            elif (element[-1]) == "-":
                upgrade_type_list.append(UpgradeType.N)
            else: 
                raise Exception("There is something wrong with the formula.")
        return upgrade_type_list

    
    def __parseFormulaTemporalOperator(self):
        c, _ = self.readUntil(["("])
        if c == "F":
            return TemporalOperator.F
        elif c == "G":
            return TemporalOperator.G
        elif c == "FG":
            return TemporalOperator.FG
        elif c == "GF":
            return TemporalOperator.GF
        elif c == "X":
            return TemporalOperator.X
        else:
            raise Exception("Unknown formula temporal operator")

    def __parseFormulaPathQuantifier(self):
        c = self.read(1)
        if c == "A":
            return PathQuantifier.A
        elif c == "E":
            return PathQuantifier.E
        else:
            raise Exception("Unknown formula path quantifier")

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
        simpleExpression = self.__convertToSimpleExpression(formulaExpression)
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
