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
    upgradeList = []
    coalitionExpression = None

    def __init__(self):
        pass

    def __str__(self):
        if self.upgradeList:
            return str(self.upgradeList) + str(self.coalitionExpression)
        else:
            return str(self.coalitionExpression)

    @property
    def agents(self):
        return self.coalitionExpression.coalitionAgents

    @property
    def expression(self):
        return self.coalitionExpression.simpleExpression

class CoalitionExpression():
    coalitionAgents = []
    simpleExpression = None

    def __init__(self):
        pass

    def __str__(self):
        if self.coalitionAgents:
            return str(self.coalitionAgents) + str(self.simpleExpression)
        else:
            return str(self.simpleExpression)

class UpgradeList():
    upgrades = None

    def __init__(self):
        pass

    def __str__(self):
        return '{' + ', '.join(map(str, self.upgrades)) + '}'

class Upgrade():
    updates = None

    def __init__(self):
        pass
    
    def test_uniform_upgrade_types(self):
        if len(set(i.upgradeType.value for i in self.updates)) != 1:
            raise Exception("Something wrong with update types within a upgrade.")

    def __str__(self):
        return '[' + ', '.join(map(str, self.updates)) + ']'
    
    @property
    def type(self):
        return self.updates[0].upgradeType

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

    def __repr__(self) -> str:
        return str(self)
class Agent():
    literal = None
    def __init__(self):
        pass
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
    
    def parseUpgradeFormula(self, formulaStr): # entry point
        self.setStr(formulaStr)

        formula = self.__parseUpgradeFormula()
        print("Parsed formula:", formula)
        return formula 

    def __parseUpgradeFormula(self):
        formula = UpgradeFormula()
        
        if self.peekNextChar() == '{':
            formula.upgradeList = self.__parseUpgradeList()

        formula.coalitionExpression = self.__parseCoalitionExpression()
        return formula

        
    def __parseCoalitionExpression(self):
        formula = CoalitionExpression()

        if self.peekNextChar() == "<":
            formula.coalitionAgents = self.__parseFormulaAgents()
        
        if self.peekNextChar() == "(":
            formula.simpleExpression = self.__parseFormulaExpression()
        else:
            formula.simpleExpression = self.__parseUpgradeFormula()

        return formula


    def __parseUpgradeList(self):
        upgrade_list = UpgradeList()
        upgrade_list.upgrades = []
        idx = self.__findMatchingParenthesis('{')
        self.consume("{")
        while self.idx < idx-1:
            upgrade_list.upgrades.append(self.__parseUpgrade())
            if self.peekNextChar() == ",":
                self.consume(",")
        self.consume("}")
        return upgrade_list


    def __parseUpgrade(self):
        upgrade = Upgrade()
        upgrade.updates = []
        idx = self.__findMatchingParenthesis('[')
        self.consume("[")
        while self.idx < idx-1:
            upgrade.updates.append(self.__parseUpdate())
            if self.peekNextChar() == ",":
                self.consume(",")
        self.consume("]")
        upgrade.test_uniform_upgrade_types()
        return upgrade

    
    def __parseUpdate(self):
        update = Update()
        self.consume("(")
        update.fromState = self.__parseUpgradeFormula()
        self.consume(",")
        update.agent = self.__parseCoalitionAgent()
        self.consume(",")
        update.toState = self.__parseUpgradeFormula()
        self.consume(")")
        update.upgradeType = self.__parseFormulaUpgradeType()

        return update

    def __parseCoalitionAgent(self):
        agent = Agent()
        res = self.readUntil(",")
        agent.literal = res[0]

        return agent


    def __parseFormulaUpgradeType(self): #  notes if the update is positive or negative.                                     
        c = self.read(1)
        if c == "+":
            return UpgradeType.P
        elif c == "-":
            return UpgradeType.N
        else:
            raise Exception("Unknown Upgrade Type")

    def __findMatchingParenthesis(self, char):
        parenthesis = {'(':')','[':']','{':'}'}
        match = parenthesis[char]
        startidx = self.idx

        self.readUntil(char)
        
        count = 0
        while True:
            if self.peekChar(0) == char:
                count += 1
            elif self.peekChar(0) == match:
                count -= 1

            self.stepForward()

            if count == 0:
                matchidx = self.idx
                self.idx = startidx

                return matchidx

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
        if self.peekChar(0) == "{":
                formulaExpression.append(self.__parseUpgradeFormula())
        elif self.peekChar(0) == "<":
                formulaExpression.append(self.__parseCoalitionExpression())
        else: 
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
