from manim import *

class Animator(Scene):
    
    def construct(self):

        text = Text("Activation Functions for Neural Networks", color=RED, font_size=16)
        self.play(Create(text))
        self.wait(1)
        
        self.play(FadeOut(text))

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 1, 1],
            x_length=6,
            y_length=3,
            axis_config={"color": GREEN},
            tips=False
        )

        x_label = Text("X", font_size=14, color=YELLOW).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_label = Text("Y", font_size=14, color=YELLOW).next_to(axes.y_axis, UP, buff=0.1)

        x_tick_values = [-3, -2, -1, 1, 2, 3]
        y_tick_values = [-1, -0.5, 0, 0.5, 1]

        x_ticks = [Text(f"{value}", font_size=12, color=YELLOW).next_to(axes.x_axis.get_tick(value), DOWN, buff=0.1) for value in x_tick_values]
        y_ticks = [Text(f"{value}", font_size=12, color=YELLOW).next_to(axes.y_axis.get_tick(value), LEFT, buff=0.1) for value in y_tick_values]
        

        functions = [
            self.binary_step(),
            self.linear(),
            self.sigmoid(),
            self.tanh(),
            self.relu(),
            self.leaky_relu(),
            self.parametrized_relu(),
            self.exponential_linear_unit(),
            self.swish(),
        ]
        names = [
            "Binary Step",
            "Linear",
            "Sigmoid",
            "Tanh",
            "ReLU",
            "Leaky ReLU",
            "Parametrized ReLU",
            "Exponential Linear Unit",
            "Swish",
        ]
        self.play(Write(x_label), Write(y_label), *[Create(tick) for tick in x_ticks], *[Create(tick) for tick in y_ticks])

        for i in range(len(functions)-1):
            if i == 0:
                graph = axes.plot(functions[i], color=RED)
                
                name = Text(names[i], font_size=14).next_to(axes.y_axis, LEFT, buff=0.1).shift(0.5*UP)
                self.play(Create(axes), Create(graph), Create(name))
                self.wait(1)
            else:
                graph2 = axes.plot(functions[i+1], color=RED)
                name2 = Text(names[i+1], font_size=14).next_to(axes.y_axis, LEFT, buff=0.1).shift(0.5*UP)
                self.play(Transform(graph, graph2), Transform(name, name2))
                self.wait(1)

        self.play(FadeOut(graph), FadeOut(name), FadeOut(axes), FadeOut(x_label), FadeOut(y_label), *[FadeOut(tick) for tick in x_ticks], *[FadeOut(tick) for tick in y_ticks])
        self.wait(0.5)

        square = Square(color=WHITE)
        text = Text("fin",color=RED, font_size=18)
        group = VGroup(square, text)
        self.play(Create(square), Create(text))
        self.wait(1)
        

    def binary_step(self):
        return lambda x: 1 if x >= 0 else 0

    def linear(self):
        return lambda x: x
    
    def sigmoid(self):
        return lambda x: 1/(1+np.exp(-x))
    
    def tanh(self):
        return lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def relu(self):
        return lambda x: np.maximum(0,x)
    
    def leaky_relu(self):
        return lambda x: np.maximum(0.1*x,x)
    
    def parametrized_relu(self):
        return lambda x,a=1: np.maximum(a*x,x)
    
    def exponential_linear_unit(self):
        return lambda x,a=1: np.maximum(a*(np.exp(x)-1),x)
    
    def swish(self):
        return lambda x,a=1: x*1/(1+np.exp(-x))
    
    def softmax(self):
        return lambda x: np.exp(x)/np.sum(np.exp(x))

Animator().render()
