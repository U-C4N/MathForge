import unittest
from dataclasses import replace, FrozenInstanceError
from unittest.mock import patch

from mathforge import Rational, symbol, Truth, rational_function, ComputationLimits
from mathforge.errors import InvalidInput, UnsupportedOperation
from mathforge.limits import computation, ResourceLimitError
from mathforge import functions as f
from mathforge import function_calculus as c
from mathforge.function_checks import equivalent


class FunctionTests(unittest.TestCase):
    def setUp(self):
        self.x,self.y=symbol("x"),symbol("y")

    def parse(self,text):return f.parse_function(text,symbols={"x":self.x})

    def checked(self,operation,args):
        with computation():
            value,evidence=c.produce(operation,args)
            self.assertTrue(c.check_operation(operation,args,value,evidence))
        return value,evidence

    def test_domains_survive_annihilation_and_zero_powers(self):
        for text in ("0*log(x)","log(x)-log(x)","log(x)**0","0/x","(1/x)**0"):
            function=self.parse(text)
            self.assertIs(function.contains({self.x:0}),Truth.FALSE,text)
            self.assertIs(function.contains({self.x:2}),Truth.TRUE,text)
        raw=0*f.log(self.x)
        self.assertEqual(raw.op,"mul")
        function=f.real_function(raw,(self.x,))
        self.assertEqual(function.expression,f.ZERO)
        self.assertIs(function.contains({self.x:-1}),Truth.FALSE)

    def test_unknown_membership_and_explicit_axes(self):
        function=self.parse("log(sin(x))")
        self.assertIs(function.contains({self.x:1}),Truth.UNKNOWN)
        with self.assertRaises(UnsupportedOperation):function.evaluate({self.x:1})
        with self.assertRaises(InvalidInput):f.real_function(f.sin(self.x),(self.y,))
        with self.assertRaises(InvalidInput):f.real_function(f.sin(self.x),(self.x,self.x))
        with self.assertRaises(FrozenInstanceError):function.variables=()

    def test_legacy_rational_holes(self):
        source=rational_function(self.x**2-1,self.x-1,variable=self.x)
        converted=f.real_function(source)
        self.assertIs(converted.contains({self.x:1}),Truth.FALSE)
        self.assertTrue(equivalent(converted.evaluate({self.x:2}),f.as_real_expr(3)))
        annihilated=0*converted
        self.assertEqual(annihilated.expression,f.ZERO)
        self.assertIs(annihilated.contains({self.x:1}),Truth.FALSE)

    def test_parser_exact_constants_and_no_execution(self):
        self.assertEqual(self.parse("0.1 + 0.2").expression,f.as_real_expr(Rational(3,10)))
        self.assertEqual(self.parse("sin(pi/6)").expression,f.as_real_expr(Rational(1,2)))
        self.assertEqual(self.parse("cos(pi)").expression,f.as_real_expr(-1))
        for text in ("__import__('os')","x.__class__","[x]","True","lambda: x","sin(x, x)","unknown(x)"):
            with self.assertRaises(InvalidInput,msg=text):self.parse(text)
        with self.assertRaises(UnsupportedOperation):self.parse("x**0.5")
        for text in ("x**(0*log(-1))","x**(0*log(x))","x**(0/x)"):
            with self.assertRaises(UnsupportedOperation):self.parse(text)

    def test_all_elementary_derivatives_independent_checker(self):
        for text in ("sin(x)","cos(x)","tan(x)","asin(x)","acos(x)","atan(x)",
                     "sinh(x)","cosh(x)","tanh(x)","exp(x)","log(x)","sqrt(x)","abs(x)",
                     "x/(1+x*x)","sin(exp(x))","x**-3"):
            with self.subTest(text=text):
                args={"expression":self.parse(text),"variable":self.x,"n":1}
                value,evidence=self.checked("function_differentiate",args)
                with computation(),patch.object(c,"_differentiate_expr",side_effect=AssertionError("producer called")):
                    self.assertTrue(c.check_operation("function_differentiate",args,value,evidence))

    def test_higher_derivatives_and_corruption(self):
        args={"expression":self.parse("sin(x)"),"variable":self.x,"n":3}
        value,evidence=self.checked("function_differentiate",args)
        self.assertTrue(equivalent(value.expression,-f.cos(self.x)))
        with computation():
            self.assertFalse(c.check_operation("function_differentiate",args,f.real_function(7,(self.x,)),evidence))
            corrupt=replace(evidence[0],derivatives=evidence[0].derivatives[:-1]+(f.real_function(7,(self.x,)),))
            self.assertFalse(c.check_operation("function_differentiate",args,f.real_function(7,(self.x,)),(corrupt,)))
            self.assertFalse(c.check_operation("function_differentiate",dict(args,expression=self.parse("cos(x)")),value,evidence))

    def test_smoothness_domain_and_partial_claim(self):
        args={"expression":self.parse("abs(x)"),"variable":self.x,"n":1}
        value,_=self.checked("function_differentiate",args)
        self.assertIs(value.contains({self.x:0}),Truth.FALSE)
        self.assertTrue(c.is_partial("function_differentiate",args,value))
        self.assertTrue(equivalent(value.evaluate({self.x:-2}),f.as_real_expr(-1)))
        square=self.parse("abs(x)**2")
        derivative=c.differentiate(square,self.x)
        self.assertIs(derivative.contains({self.x:0}),Truth.TRUE)
        restricted=self.parse("sqrt(x)**2")
        derivative=c.differentiate(restricted,self.x)
        self.assertIs(derivative.contains({self.x:0}),Truth.FALSE)
        for operation,args in (
            ("function_gradient",{"expression":restricted,"variables":(self.x,)}),
            ("function_hessian",{"expression":restricted,"variables":(self.x,)}),
            ("function_jacobian",{"expressions":(restricted,),"variables":(self.x,)}),
        ):
            value,_=self.checked(operation,args)
            self.assertTrue(c.is_partial(operation,args,value))

    def test_ordered_gradient_jacobian_hessian(self):
        source=f.real_function(f.sin(self.x)*f.exp(self.y),(self.x,self.y))
        for operation,args in (
            ("function_gradient",{"expression":source,"variables":(self.y,self.x)}),
            ("function_jacobian",{"expressions":(source,source*2),"variables":(self.x,self.y)}),
            ("function_hessian",{"expression":source,"variables":(self.x,self.y)}),
        ):
            value,evidence=self.checked(operation,args)
            with computation():self.assertFalse(c.check_operation(operation,args,value,evidence[::-1]))
        gradient=c.gradient(source,(self.y,self.x))
        self.assertTrue(equivalent(gradient[0].expression,f.sin(self.x)*f.exp(self.y)))

    def test_taylor_coefficients_and_corruption(self):
        for text,expected in (
            ("exp(x)",(1,1,Rational(1,2),Rational(1,6),Rational(1,24))),
            ("sin(x)",(0,1,0,Rational(-1,6),0)),
            ("log(1+x)",(0,1,Rational(-1,2),Rational(1,3),Rational(-1,4))),
        ):
            args={"expression":self.parse(text),"variable":self.x,"point":0,"order":4}
            value,evidence=self.checked("function_taylor",args)
            self.assertTrue(all(equivalent(a,f.as_real_expr(b)) for a,b in zip(value.coefficients,expected)))
            corrupt=replace(value,coefficients=(f.as_real_expr(17),)+value.coefficients[1:])
            with computation():self.assertFalse(c.check_operation("function_taylor",args,corrupt,evidence))
        with self.assertRaises(UnsupportedOperation):c.taylor(self.parse("abs(x)"),self.x,0,3)
        with self.assertRaises(UnsupportedOperation):c.taylor(self.parse("sqrt(x)"),self.x,0,3)
        with self.assertRaises(UnsupportedOperation):c.taylor(self.parse("sin(x)"),self.x,f.pi,3)
        with self.assertRaises(InvalidInput):c.taylor(self.parse("sin(x)"),self.x,0*f.log(-1),3)

    def test_elementary_catalogue_and_deferred(self):
        for text in ("sin(2*x)","cos(x)","exp(x)","sinh(x)","cosh(x)","tan(x)","tanh(x)",
                     "log(x)","sqrt(x)","abs(x)","1/x","1/(1+x*x)","x*exp(x)","x*sin(x)","x*x*cos(x)",
                     "asin(x)","acos(x)","atan(x)","x*x*log(2*x+1)","2*x*cos(x*x)","exp(x/(1+x))/(1+x)**2"):
            with self.subTest(text=text):self.checked("function_antiderivative",{"expression":self.parse(text),"variable":self.x})
        value=c.antiderivative(self.parse("exp(-x*x)"),self.x)
        self.assertIsInstance(value,f.UnevaluatedIntegral)

    def test_finite_and_infinite_limits_with_corruption(self):
        scenarios=(
            ("sin(x)/x",0,"both","finite",1),
            ("(exp(x)-1)/x",0,"both","finite",1),
            ("log(1+x)/x",0,"both","finite",1),
            ("1/x",0,"both","does_not_exist",None),
            ("log(x)",0,"right","negative_infinity",None),
            ("sqrt(x)",0,"right","finite",0),
            ("exp(x)/x**5","+inf","both","positive_infinity",None),
            ("x**3*exp(-x)","+inf","both","finite",0),
            ("log(x)**3/x","+inf","both","finite",0),
            ("(exp(x)+1)/(exp(x)-1)","+inf","both","finite",1),
            ("(x+1)/(x*x+1)","+inf","both","finite",0),
            ("log(log(x))","+inf","both","positive_infinity",None),
            ("sin(x)/x","+inf","both","finite",0),
            ("sin(x)","+inf","both","does_not_exist",None),
            ("x*log(x)",0,"right","finite",0),
            ("cos(3*x+1)","-inf","both","does_not_exist",None),
            ("(1+sin(x))*exp(-x)","+inf","both","finite",0),
            ("x*x*log(-x)",0,"left","finite",0),
        )
        for text,point,side,kind,expected in scenarios:
            with self.subTest(text=text):
                args={"expression":self.parse(text),"variable":self.x,"point":point,"side":side}
                value,evidence=self.checked("function_limit",args)
                self.assertEqual(value.kind,kind)
                if expected is not None:self.assertTrue(equivalent(value.value,f.as_real_expr(expected)))
                with computation(),patch.object(c,"limit",side_effect=AssertionError("producer called")):
                    self.assertTrue(c.check_operation("function_limit",args,value,evidence))
                corrupt=c.FunctionLimitValue("finite",f.as_real_expr(73))
                certificate=replace(evidence[0],value=corrupt)
                with computation():self.assertFalse(c.check_operation("function_limit",args,corrupt,(certificate,)))

    def test_unknown_limit_and_excluded_approach_defer(self):
        self.assertIsInstance(c.limit(self.parse("sin(exp(x))"),self.x,"+inf"),f.UnevaluatedLimit)
        self.assertIsInstance(c.limit(self.parse("log(x)"),self.x,0,side="left"),f.UnevaluatedLimit)
        restricted=f.real_function(self.x,(self.x,),domain=f.condition("positive",-f.as_real_expr(self.x)))
        self.assertIsInstance(c.limit(restricted,self.x,"+inf"),f.UnevaluatedLimit)
        with self.assertRaises(InvalidInput):self.parse("sin(x)").evaluate({self.x:self.y})

    def test_limit_check_does_not_invoke_any_producer(self):
        from mathforge import function_limits as fl
        args={"expression":self.parse("sin(x)/x"),"variable":self.x,"point":0,"side":"both"}
        value,evidence=self.checked("function_limit",args)
        with computation(),patch.object(fl,"_produce",side_effect=AssertionError("producer called")),patch.object(c,"_differentiate_expr",side_effect=AssertionError("producer called")):
            self.assertTrue(c.check_operation("function_limit",args,value,evidence))
        for text,point,side in (("log(x)**2/exp(x)","+inf","both"),("sin(x)/x","+inf","both"),
                                ("sin(x)","+inf","both"),("x*log(x)",0,"right")):
            args={"expression":self.parse(text),"variable":self.x,"point":point,"side":side}
            value,evidence=self.checked("function_limit",args)
            with computation(),patch.object(fl,"_produce",side_effect=AssertionError("producer called")),patch.object(c,"_scale",side_effect=AssertionError("producer called")):
                self.assertTrue(c.check_operation("function_limit",args,value,evidence))

    def test_adversarial_domains_and_integer_signs(self):
        self.assertIs(f.condition("positive",f.log(-1)).contains(),Truth.FALSE)
        self.assertIs(f.condition("positive",f.exp(f.log(-1))).contains(),Truth.FALSE)
        guard=f.condition("nonnegative",(1/f.as_real_expr(self.x))**2)
        self.assertIs(guard.contains({self.x:0}),Truth.FALSE)
        self.assertIs(guard.contains({self.x:2}),Truth.TRUE)
        for text,expected in (("log(2)**-3",1),("(-2)**-3",-1),("(-2)**-2",1),("0",0)):
            sign=f.sign_of(self.parse(text).expression)
            self.assertIs(type(sign),int)
            self.assertEqual(sign,expected)
        # An unknown transcendental sign remains unknown, not a numerical guess.
        self.assertIsNone(f.sign_of(self.parse("sin(1)").expression))
        source=self.parse("abs(x)")
        args={"expression":source,"variable":self.x,"n":1}
        value,evidence=self.checked("function_differentiate",args)
        forged=f.RealFunction(value.expression,(self.x,),f.TRUE)
        # The direct constructor retains its intrinsic divisor restriction too.
        self.assertIs(forged.contains({self.x:0}),Truth.FALSE)
        derivative=f.RealFunction(f.ONE,(self.x,),f.TRUE)
        trace=replace(evidence[0],derivatives=(derivative,))
        with computation():self.assertFalse(c.check_operation("function_differentiate",args,derivative,(trace,)))
        self.assertIs(f.TRUE.contains({self.x:f.log(-1)}),Truth.FALSE)
        with self.assertRaises(InvalidInput):self.parse("x").evaluate({self.x:f.log(-1)})
        with self.assertRaises(UnsupportedOperation):self.parse("x").evaluate({self.x:f.sqrt(f.sin(1))})
        self.assertFalse(equivalent(0*f.log(-1),f.ZERO))
        args={"expression":self.parse("x"),"variable":self.x,"point":0,"side":"both"}
        value,evidence=self.checked("function_limit",args)
        forged=c.FunctionLimitValue("finite",0*f.log(-1))
        with computation():self.assertFalse(c.check_operation("function_limit",args,forged,(replace(evidence[0],value=forged),)))
        with self.assertRaises(InvalidInput):c.limit(self.parse("x"),self.x,0*f.log(-1))
        with computation():self.assertFalse(c.check_operation("function_limit",dict(args,point=0*f.log(-1)),value,evidence))

    def test_nested_endpoint_and_asymptotic_signs(self):
        for text,point,side,kind,expected in (
            ("log(sqrt(x))",0,"right","negative_infinity",None),
            ("exp(-x*x)","+inf","both","finite",0),
            ("-exp(x)","+inf","both","negative_infinity",None),
            ("exp(x*x)-exp(x)","+inf","both","positive_infinity",None),
            ("log(-x)","-inf","both","positive_infinity",None),
            ("log(x*x)-2*log(x)","+inf","both","finite",0),
        ):
            args={"expression":self.parse(text),"variable":self.x,"point":point,"side":side}
            value,_=self.checked("function_limit",args)
            self.assertEqual(value.kind,kind)
            if expected is not None:self.assertTrue(equivalent(value.value,f.as_real_expr(expected)))
    def test_computation_budgets(self):
        with self.assertRaises(ResourceLimitError):
            c.differentiate(self.parse("sin(exp(x))"),self.x,50,limits=ComputationLimits(max_work=20))
        with self.assertRaises(ResourceLimitError):
            f.parse_function("x**100000",symbols={"x":self.x},limits=ComputationLimits(max_work=5))


if __name__=="__main__":unittest.main()
