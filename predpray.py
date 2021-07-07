from manimlib import *

def get_vecfield_preypred(alpha, beta, gamma, delta):
    def v(x, y):
        xdot = alpha * x - beta * x * y
        ydot = -gamma * y + delta * x * y
        return np.array([xdot, ydot, 0.0])
    return v

def get_vecfield_toxiclove(k, m):
    def v(x, y):
        xdot = k * y
        ydot = -1 * m * x
        return np.array([xdot, ydot, 0.0])
    return v

def get_vecfield_toxicbutcarefullove(k, m, n):
    def v(x, y):
        xdot = k * y
        ydot = -1 * m * x - n * y
        return np.array([xdot, ydot, 0.0])
    return v

def get_vecfield_lorenzattractor(sigma, beta, rho):
    def v(x, y, z):
        xdot = sigma * (y - x)
        ydot = x * (rho - z) - y
        zdot = x * y - beta * z

        return np.array([xdot, ydot, zdot])

    return v

class AgentBasedSim(VGroup):
    CONFIG = {
        "cell_size" : 0.05,
        "box_size" : 7,
        "update_frequency" : 0.05,
        "reproduction_prob" : 0.09,
        "predator_death_prob" : 0.1,
        "predator_birth_prob" : 0.9,
        "initial_prey" : 0.1,
        "initial_pred" : 0.01
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim = int(self.box_size // self.cell_size)
        self.time = 0.0
        self.last_update_time = -np.inf
        self.isupdating = False
        self.initialize_population()
        self.setup_grid()

        #updaters
        self.add_updater(lambda m, dt : m.update_time(dt))
        self.add_updater(lambda m, dt : m.evolve(dt))

    def initialize_population(self):
        self.population = np.zeros((self.dim, self.dim), dtype=int)
        num_preys = int(self.initial_prey * self.dim * self.dim)
        num_preds = int(self.initial_pred * self.dim * self.dim)

        count = 0
        while count < num_preys:
            i = np.random.randint(self.dim)
            j = np.random.randint(self.dim)

            if self.population[i, j] == 0:
                count += 1
                self.population[i, j] = 1

        count = 0
        while count < num_preds:
            i = np.random.randint(self.dim)
            j = np.random.randint(self.dim)

            if self.population[i, j] == 0:
                count += 1
                self.population[i, j] = -1

    def get_index(self, i, j):
        return i + self.dim * j
    
    def set_status(self, i, j, state):
        index = self.get_index(i, j)
        if state == 0:
            self.grid[index].set_fill(BLACK)
            self.grid[index].set_stroke(BLACK, width=0)
        if state == 1:
            self.grid[index].set_fill(BLUE).set_opacity(0.8)
            self.grid[index].set_stroke(BLACK, width=0)
        if state == -1:
            self.grid[index].set_fill(RED).set_opacity(0.8)
            self.grid[index].set_stroke(BLACK, width=0)

    def update_boxes(self):
        for i in range(self.dim):
            for j in range(self.dim):
                self.set_status(i, j, self.population[i, j])

    def setup_grid(self):
        self.grid = Square(side_length=self.cell_size, stroke_width=1.0, stroke_opacity=0.2).get_grid(self.dim, self.dim, self.box_size, buff=0)
        self.update_boxes()
        self.add(self.grid, Square(side_length=self.box_size, stroke_width=1.0))

    def update_time(self, dt):
        if self.isupdating:
            self.time += dt

    def pauseorresume(self, val):
        self.isupdating = val

    def get_counts(self):
        preds = 0
        preys = 0
        for i in range(self.dim):
                for j in range(self.dim):
                    if self.population[i, j] == 1:
                        preys += 1
                    elif self.population[i, j] == -1:
                        preds += 1

        return np.array([preys, preds])

    def get_normalised_counts(self):
        return self.get_counts()/(self.dim * self.dim)

    def get_cells_cooresponding_to_state(self, state):
        result = VGroup()
        for i in range(self.dim):
            for j in range(self.dim):
                if self.population[i, j] == state:
                    result.add(self.grid[self.get_index(i, j)])

        return result

    def evolve(self, dt):
        if self.isupdating:
            if (self.time - self.last_update_time) > self.update_frequency:
                self.last_update_time = self.time

                for i in range(self.dim):
                    for j in range(self.dim):
                        curr = self.population[i, j]
                        #predator
                        if curr == -1:
                            ground_indices = []
                            meal_indices = []
                            for x in range(-1, 2):
                                for y in range(-1, 2):
                                    if 0 <= i + x < self.dim and 0 <= j + y < self.dim:
                                        #check if ground
                                        if self.population[i+x, j+y] == 0:
                                            ground_indices.append([i+x, j+y])
                                        #check if meal
                                        elif self.population[i+x, j+y] == 1:
                                            meal_indices.append([i+x, j+y])
                            
                            #if there is meal, eat and procreate!!
                            if len(meal_indices) > 0:
                                choice = random.choice(meal_indices)
                                if random.random() < self.predator_birth_prob:
                                    self.population[choice[0], choice[1]] = -1
                                    #move if there is space
                                if len(ground_indices) > 0:
                                    choice = random.choice(ground_indices)
                                    self.population[choice[0], choice[1]] = -1
                                    self.population[i, j] = 0
                            #no meals == death!
                            else:
                                if random.random() < self.predator_death_prob:
                                    self.population[i, j] = 0
                                elif len(ground_indices) > 0:
                                    choice = random.choice(ground_indices)
                                    self.population[choice[0], choice[1]] = -1
                                    self.population[i, j] = 0
                        #prey
                        if curr == 1:
                            ground_indices = []
                            for x in range(-1, 2):
                                for y in range(-1, 2):
                                    if 0 <= i + x < self.dim and 0 <= j + y < self.dim:
                                        #check if ground
                                        if self.population[i+x, j+y] == 0:
                                            ground_indices.append([i+x, j+y])
                            
                            #if there is ground, procreate!!
                            if len(ground_indices) > 0:
                                choice = random.choice(ground_indices)
                                if random.random() < self.reproduction_prob:
                                    self.population[choice[0], choice[1]] = 1
                                    ground_indices.remove(choice)
                            #move if there is space
                            if len(ground_indices) > 0:
                                choice = random.choice(ground_indices)
                                self.population[choice[0], choice[1]] = 1
                                self.population[i, j] = 0
                    
                self.update_boxes()

class AgentBasedSimGraph(VGroup):
    CONFIG = {
        "update_freq" : 1/60.0
    }

    def __init__(self, sim, **kwargs):
        super().__init__(**kwargs)
        self.sim = sim
        self.time = 0.0
        self.last_update_time = -np.inf
        self.initialise_counts()
        self.setup_axes()
        self.setup_graph()
        self.add_ticks()
        self.add_legends()

        self.add_updater(lambda m, dt: m.update_time(dt))
        self.add_updater(lambda m, dt : m.update_graphs(dt))
        self.add_updater(lambda m, dt : m.update_ticks(dt))

    def initialise_counts(self):
        self.counts = [self.sim.get_normalised_counts()]

    def setup_axes(self):
        axes = Axes((0, 1.0), (0.0, 1.1, 0.2), height=4, width=4, y_axis_config = {
                "tick_frequency" : 0.1
            })
        axes.add_coordinate_labels(x_values= [], y_values=np.arange(0.0, 1.1, 0.2), num_decimal_places=1, font_size=16)
        self.axes = axes
        self.add(self.axes)

    def add_ticks(self):
        self.x_ticks = VGroup()
        self.x_labels = VGroup()
        self.add(self.x_ticks, self.x_labels)

    def add_legends(self):
        pred = VGroup(Line(LEFT, RIGHT).set_width(0.3).set_color(RED), Text("Predator").scale(0.4)).arrange(buff=0.1)
        prey = VGroup(Line(LEFT, RIGHT).set_width(0.3).set_color(BLUE), Text("Prey").scale(0.4)).arrange(buff=0.1)
        pred.move_to(self.axes.c2p(1.1, 1))
        prey.next_to(pred, DOWN, aligned_edge=LEFT)
        self.add(pred, prey)

    def setup_graph(self):
        self.graph = self.get_graph()
        self.add(self.graph)

    def update_time(self, dt):
        if self.sim.isupdating:
            self.time += dt

    def get_graph(self):
        axes = self.axes
        counts = self.counts

        preds = []
        preys = []
        for x, counts in zip(np.linspace(0.0, 1.0, len(counts)), counts):
            prey = axes.c2p(x, counts[0])
            pred = axes.c2p(x, counts[1])
            preds.append(pred)
            preys.append(prey)

        prey_line = VGroup()
        for i in range(len(preys)-1):
            prey_line.add(Line(preys[i], preys[i+1], color=BLUE, stroke_width=4.0))

        pred_line = VGroup()
        for i in range(len(preds)-1):
            pred_line.add(Line(preds[i], preds[i+1], color=RED, stroke_width=4.0))

        region = VGroup(pred_line, prey_line)
        return region

    def update_ticks(self, dt):
        if self.sim.isupdating:
            tick_height = 0.03 * self.get_height()
            tick_template = Line(DOWN, UP).set_height(tick_height)

            if 0 < self.time < 10:
                tick_range = range(0, int(self.time)+1, 1)
            elif self.time < 50:
                tick_range = range(5, int(self.time)+1, 5)
            elif self.time < 100:
                tick_range = range(10, int(self.time)+1, 10)
            else:
                tick_range = range(20, int(self.time)+1, 20)

            def get_tick(x):
                tick = tick_template.copy()
                tick.move_to(self.axes.c2p(x/self.time, 0))
                return tick

            def get_label(x, tick):
                label = Integer(x)
                label.set_height(tick_height)
                label.next_to(tick, DOWN, buff=0.2*tick_height)
                return label

            x_ticks = VGroup()
            x_labels = VGroup()
            for x in tick_range:
                tick = get_tick(x)
                x_ticks.add(tick)
                x_labels.add(get_label(x, tick))

            self.x_ticks.become(x_ticks)
            self.x_labels.become(x_labels)

    def get_data(self):
        return self.sim.get_normalised_counts()

    def update_graphs(self, dt):
        if self.sim.isupdating:
            if (self.time - self.last_update_time) > self.update_freq:
                self.last_update_time = self.time
                self.counts.append(self.get_data())
                self.graph.become(self.get_graph())

class LorenzQuote(Scene):

    def construct(self):

        quote = Text(
            '''
            Chaos: When the present determines the future, 

            but the approximate present does not 

            approximately determine the future


                                        - Edward Lorenz
            ''',
        t2c = {
            "Chaos" : RED,
            "Edward Lorenz" : BLUE
        },
        t2s = {
            "Edward Lorenz" : ITALIC
        }
        )
        quote.scale(0.6)
        self.play(
            ShowIncreasingSubsets(quote), run_time=4.0
        )
        # self.add(quote)

class Chapters(Scene):

    def construct(self):
        screen = FullScreenRectangle()
        screen.set_color(GREY_D)
        self.add(screen)
        chap1 = ScreenRectangle().set_color(BLACK).set_opacity(1.0).set_width(5)
        chap2 = ScreenRectangle().set_color(BLACK).set_opacity(1.0).set_width(5)
        chap3 = ScreenRectangle().set_color(BLACK).set_opacity(1.0).set_width(5)
        chap4 = ScreenRectangle().set_color(BLACK).set_opacity(1.0).set_width(5)
        rects = VGroup(chap1, chap2, chap3, chap4).arrange_in_grid(2, 2, buff=1.5)
        part1 = TexText("Chapter 1")
        part1.move_to(chap1)
        part2 = TexText("Chapter 2")
        part2.move_to(chap2)
        part3 = TexText("Chapter 3")
        part3.move_to(chap3)
        part4 = TexText("Chapter 4")
        part4.move_to(chap4)
        self.add(rects, part1, part2, part3, part4)
        self.wait(3)
        frame = self.camera.frame
        frame.save_state()
        self.play(
        frame.animate.replace(chap1)
        )
        self.wait()

        quote = Text(
            '''
            Chaos: When the present determines the future, 

            but the approximate present does not 

            approximately determine the future


            - Edward Lorenz
            ''',
        t2c = {
            "Chaos" : RED,
            "Edward Lorenz" : BLUE
        },
        t2s = {
            "Edward Lorenz" : ITALIC
        }
        )
        quote.move_to(chap1)
        quote.scale(0.2)
        self.play(
            FadeTransform(part1, quote)
        )
        self.wait()
        present = quote.get_part_by_text("present")
        future = quote.get_part_by_text("future")
        self.play(
            present.animate.set_color(GREEN)
        )
        self.wait()
        self.play(
            future.animate.set_color(YELLOW)
        )
        self.wait()


class LotkaVolterra(Scene):
    CONFIG = {
        "random_seed" : 2
    }
    
    def construct(self):
        model = AgentBasedSim(box_size=7)
        graph = AgentBasedSimGraph(model)
        model.to_edge(LEFT)
        graph.next_to(model, RIGHT, aligned_edge=DOWN)

        self.play(
            FadeInFromPoint(model, ORIGIN)
        )
        self.play(
            FadeInFromPoint(graph, ORIGIN)
        )
        model.pauseorresume(True)
        # self.add(model, graph)

        lotkavolterra = Text("Lotka - Volterra Model",
        font_size=24)
        predprey = Text("Prey - Predator Model",
        t2c={"Prey" : BLUE, "Predator":RED},
        font_size=24
        )
        lotkavolterra.next_to(graph, UP, buff=2.0)
        predprey.move_to(lotkavolterra)

        self.wait(5)
        self.play(
            Write(lotkavolterra),
        )
        self.wait()
        self.play(
            FadeTransform(lotkavolterra, predprey)
        )
        self.play(
            ShowCreation(Underline(predprey))
        )

        self.wait(3)
        rabbit = Square(side_length=0.5).set_color(BLUE).set_opacity(0.8)
        fox = Square(side_length=0.5).set_color(RED).set_opacity(0.8)
        rabbit.next_to(predprey, DOWN).shift(LEFT)
        fox.next_to(predprey, DOWN).shift(RIGHT)
        self.play(
            GrowFromCenter(rabbit)
        )
        self.wait()
        self.play(
            GrowFromCenter(fox)
        )

        grass = ImageMobject("grass.png")
        grass.scale(0.3)
        grass.next_to(predprey, DOWN)
        # grass.rotate(PI)
        self.wait()
        self.play(
            ShowCreation(grass)
        )

        self.wait()
        self.play(
            FadeOutToPoint(grass, rabbit.get_center())
        )
        self.wait()
        self.play(
            FadeOutToPoint(rabbit, fox.get_center())
        )
        self.wait(2)
        self.play(
            FadeOut(fox)
        )
        self.wait()
        howdowemodel = Text(
            """
            How do we model this system??
            """,
            font_size=20
        )
        howdowemodel.next_to(predprey, DOWN, buff=0.5)
        self.play(
            Write(howdowemodel)
        )
        self.wait(20)

class OnlyRabbits(Scene):

    def construct(self):
        pp = AgentBasedSim(box_size=5, initial_pred=0.0, initial_prey=0.01)
        pp.to_edge(LEFT)
        ppgraph = AgentBasedSimGraph(pp)
        ppgraph.next_to(pp, RIGHT, buff=2.0)
        # self.add(pp, ppgraph)
        self.play(
            ShowCreation(pp)
        )
        self.wait(2)
        rabbits = pp.get_cells_cooresponding_to_state(1)
        self.play(
            FlashAround(pp)
        )
        self.wait()
        self.play(
            ShowCreation(ppgraph)
        )
        pp.pauseorresume(True)
        self.wait(4)
        pp.pauseorresume(False)
        self.play(
            FadeOut(ppgraph),
            FadeOut(pp)
        )
        pp = self.get_pop()
        self.wait()
        self.play(
            FadeIn(pp)
        )
        # self.add(pp)
        rabbits = pp.get_cells_cooresponding_to_state(1)
        varR = Tex("R", "({t})", tex_to_color_map = {"{t}" : GREY, "R" : BLUE})
        varR.shift(RIGHT * 3 + UP)

        self.wait(2)
        self.play(
            TransformFromCopy(rabbits, varR[0])
        )
        self.wait()

        whatisR = Text("Number of rabbits in the population", font_size=24)
        whatisR.next_to(varR, UP)

        self.play(
            Write(whatisR)
        )

        self.wait(3)

        Rismass = Text("The total mass of rabbits", font_size=24)
        Rismass.next_to(varR, UP)

        self.play(
            FadeTransform(whatisR, Rismass)
        )

        self.wait(3)

        self.play(
            FadeOut(Rismass)
        )
        self.wait(3)


        self.play(
            Write(varR[1:])
        )
        varnextR = Tex("R", "({t} + \\delta {t})", tex_to_color_map = {"{t}" : GREY, "R" : BLUE})
        varnextR.next_to(varR, DOWN, buff=1.5)
        toArrow = Arrow(varR.get_center(), varnextR.get_center())
        self.wait(3)
        self.play(
            GrowArrow(toArrow),
            FadeInFromPoint(varnextR, varR.get_center())
        )
        self.wait(2)

        delR = Tex("{R}({t} + \\delta {t}) - {R}({t})", tex_to_color_map = {"{t}" : GREY, "R" : BLUE}, isolate=["+"]).shift(RIGHT*2)
        self.play(
            TransformMatchingShapes(VGroup(varR, varnextR, toArrow), delR)
        )
        self.wait()
        
        pp.pauseorresume(True)
        self.wait(2)
        pp.pauseorresume(False)

        themorewewait = Text(
            """
            The more we wait, 

            the more the number of new rabbits
            """,
            font_size=24
        ).next_to(delR, UP).shift(RIGHT + UP * 2)
        self.play(
            Write(themorewewait)
        )
        self.wait(2)
        # self.play(
        #     Transform(pp, self.get_pop())
        # )
        proportionalsym = Tex("\\propto").next_to(delR)
        delTRT = Tex("\\delta {t}",  "{R}({t})", tex_to_color_map = {"{t}" : GREY, "{R}" : BLUE}, ).next_to(proportionalsym)
        self.wait()
        self.play(
            FlashAround(delR[4:6])
        )
        self.play(
            Write(proportionalsym),
            Write(delTRT[0]),
            Write(delTRT[1])
        )

        self.wait(3)
        self.play(
            Transform(pp, self.get_pop())
        )
        self.wait()
        self.play(
            Transform(pp, self.get_pop(0.1))
        )

        themorewehave = Text(
            """
            The more we start with, 
            
            the more the number of new rabbits
            """,
            font_size=24
        ).next_to(delR, UP).shift(RIGHT + UP * 2)

        self.play(
            FadeTransform(themorewewait, themorewehave)
        )

        self.wait(2)
        self.play(
            Write(delTRT[2:])
        )
        self.wait(2)
        self.play(
            FadeOut(themorewehave)
        )
        self.wait(2)
        rhs = Tex("=", "k \\,", "\\delta {t} \\,", "{R}({t})", tex_to_color_map = {"{t}" : GREY, "{R}" : BLUE}).next_to(delR)
        self.play(
            TransformMatchingShapes(VGroup(proportionalsym, delTRT), rhs)
        )
        self.wait()
        self.play(
            FlashAround(rhs[1])
        )
        self.play(
            FlashAround(rhs[1])
        )

        #sanity check!
        self.wait(3)
        sanity = Tex("{R}({t} + 0) - {R}({t})", "=", "k \\cdot 0 \\cdot {R}({t})", tex_to_color_map = {"{t}" : GREY, "{R}" : BLUE})
        sanity.next_to(delR, DOWN, aligned_edge=LEFT)
        self.play(
            Write(sanity)
        )
        checkmark = Checkmark().next_to(sanity, LEFT)
        self.wait(2)
        self.play(
            Write(checkmark)
        )
        self.wait()
        self.play(
            FadeOut(checkmark),
            FadeOut(sanity)
        )

        delRdelt = Tex("{ {R}({t} + \\delta{t}) - {R}({t}) \\over \\delta {t} }", "=", "{k} {R}({t})", tex_to_color_map = {"{t}" : GREY, "{R}" : BLUE})
        delRdelt.shift(RIGHT * 2)
        self.wait(3)
        self.play(
            TransformMatchingShapes(VGroup(delR, rhs), delRdelt)
        )
        self.wait(3)
        dRdt = Tex("{ d {R}({t}) \\over d {t} }", "=", "{k} {R}({t})", tex_to_color_map = {"{t}" : GREY, "{R}" : BLUE})
        dRdt.shift(RIGHT * 2)
        self.play(
            TransformMatchingTex(delRdelt, dRdt)
        )
        self.wait(3)
        self.play(
            FlashAround(dRdt.get_part_by_tex("\\over"))
        )
        self.wait(5)

    def get_pop(self, prey=0.01):
        pp = AgentBasedSim(box_size=5, initial_pred=0.0, initial_prey=prey).to_edge(LEFT)
        return pp
    
class RabbitSolution(Scene):

    def construct(self):
        eqn = Tex("{d{R}({t}) \\over d{t}} = k {R}({t})", tex_to_color_map = {"{R}" : BLUE, "{t}" : GREY})
        eqn.scale(1.5)
        self.play(
            Write(eqn)
        )
        de = Tex("{d{R}({t}) \\over d{t}} = {R}({t})", tex_to_color_map = {"{R}" : BLUE, "{t}" : GREY})
        self.wait()
        self.play(
            TransformMatchingTex(eqn, de)
        )
        self.wait(2)
        self.play(
            FlashAround(de.get_part_by_tex("\\over"))
        )
        self.wait(2)
        self.play(
            de.animate.to_corner(UR)
        )

        #the number line
        nl = NumberLine((0, 2500), unit_size=1.0)
        rightpt = nl.p2n(FRAME_X_RADIUS * RIGHT)
        nl.add_numbers(x_values=range(0, int(rightpt), 1))
        nl.next_to(LEFT * 6)

        #the value tracker for time
        t = ValueTracker(0.0)
        t.add_updater(lambda m, dt : m.increment_value(dt * 0.5))

        #time
        T = VGroup(TexText("t = ", tex_to_color_map = {"t" : GREY}), DecimalNumber(0)).arrange()
        T.to_edge(LEFT).shift(UP*3)
        T.add_updater(lambda m : m[1].set_value(t.get_value()))

        #value tracker for R(t)
        Rt = ValueTracker(1.0)
        Rt.add_updater(lambda m : m.set_value(np.exp(t.get_value())))

        Rvalue = VGroup(TexText("{R}({t}) = ", tex_to_color_map = {"{t}" : GREY, "{R}" : BLUE}), DecimalNumber(1)).arrange()
        Rvalue.next_to(T, DOWN, buff=0.3, aligned_edge=LEFT)
        Rvalue.add_updater(lambda m : m[1].set_value(Rt.get_value()))

        #dot which tracks R(t)
        Rtpos = Dot(color=BLUE)
        Rtpos.add_updater(lambda m : m.move_to(nl.n2p(Rt.get_value())))

        #line to track the value of R(t)
        RtLine = Line(nl.n2p(0), nl.n2p(1), stroke_width=6.0).set_color(BLUE)
        RtLine.add_updater(lambda m : m.put_start_and_end_on(nl.n2p(0), Rtpos.get_center()))

        #the arrow which tracks rdot
        rdot = Arrow(LEFT, RIGHT)
        rdot.set_color(YELLOW)
        def rdot_updater(m):
            l = get_norm(nl.n2p(Rt.get_value()) - nl.n2p(0))
            m.set_width(l)
            m.move_to(Rtpos.get_center() + l * RIGHT * 0.5)
        rdot.add_updater(rdot_updater)

        R = Tex("R").scale(0.6).set_color(BLUE)
        dR = Tex("d{R} \\over d{t}", tex_to_color_map = {"{R}" : BLUE, "{t}" : GREY}).scale(0.6)
        always(R.next_to, RtLine, UP)
        always(dR.next_to, rdot, UP)

        # self.add(nl, Rt, t, Rtpos, rdot, RtLine, R, dR, T)
        self.wait(3)
        self.play(
            ShowCreation(nl),
            run_time=2.0
        )
        # self.add(nl)
        self.wait(2)
        self.play(
            Write(T)
        )
        self.wait(2)
        self.play(
            GrowArrow(RtLine),
            run_time=2.0
        )
        self.play(
            GrowFromCenter(Rtpos)
        )
        self.play(
            Write(R),
            Write(Rvalue)
        )
        temp_arr = Arrow(LEFT, RIGHT).set_color(YELLOW).shift(UP)
        temp_arr.set_width(1)

        self.wait(3)
        self.play(
            TransformFromCopy(de.get_part_by_tex("\\over"), temp_arr)
        )
        self.wait()
        self.play(
            temp_arr.animate.flip()
        )
        self.wait()
        self.play(
            temp_arr.animate.flip()
        )
        self.wait(2)
        self.play(
            ApplyMethod(temp_arr.scale, 3), rate_func=there_and_back, run_time=2.5
        )
        self.wait(3)
        temp_eq = Tex("{d{R}({0}) \\over d{t} } = {R}({0})", tex_to_color_map = {"{R}" : BLUE, "{0}" : GREY, "{t}" : GREY})
        temp_eq.scale(0.7).shift(DOWN)
        self.play(
            Write(temp_eq)
        )
        
        self.wait()
        self.play(
            temp_arr.animate.move_to(nl.n2p(0) + RIGHT*0.5 + DOWN*0.2)
        )
        self.wait(2)
        self.play(
            temp_arr.animate.move_to(rdot)
        )
        self.add(rdot)
        self.remove(temp_arr)
        self.play(
            FadeOut(temp_eq)
        )
        self.wait(2)

        self.play(
            Write(dR)
        )
        self.wait(3)
        self.add(t, Rt)

        self.wait_until(
            lambda: rdot.get_right()[0] > FRAME_X_RADIUS
        )
        nl.remove(nl.numbers)
        self.play(
            nl.animate.scale(0.1, about_point=nl.n2p(0))
        )
        nl.add_numbers(x_values=range(0, 201, 15), font_size=18),
        self.wait_until(
            lambda: rdot.get_right()[0] > FRAME_X_RADIUS
        )
        nl.remove(nl.numbers)
        self.play(
            nl.animate.scale(0.2, about_point=nl.n2p(0))
        )
        nl.add_numbers(x_values=range(0, 701, 50), font_size=16)
        self.wait_until(
            lambda: rdot.get_right()[0] > FRAME_X_RADIUS
        )
        nl.remove(nl.numbers)
        self.play(
            nl.animate.scale(0.3, about_point=nl.n2p(0))
        )
        nl.add_numbers(x_values=range(0, 2001, 150), font_size=16)
        self.wait_until(
            lambda: rdot.get_right()[0] > FRAME_X_RADIUS
        )

        T.clear_updaters()
        Rvalue.clear_updaters()
        self.play(
            FadeOut(dR),
            FadeOut(R),
            FadeOut(RtLine),
            FadeOut(rdot),
            FadeOut(nl),
            FadeOut(Rtpos),
            FadeOut(T),
            FadeOut(Rvalue)
        )

        #dRdt = -R
        self.wait(2)
        self.play(
            de.animate.move_to(ORIGIN)
        )
        de2 = Tex("{d{R}({t}) \\over d{t}} = -{R}({t})", tex_to_color_map={"{R}" : BLUE, "{t}" : GREY})
        self.wait()
        self.play(
            TransformMatchingTex(de, de2)
        )
        self.wait(3)

        nl = NumberLine((0, 5), unit_size=2)

        #the value tracker for time
        t = ValueTracker(0.0)
        t.add_updater(lambda m, dt : m.increment_value(dt * 0.5))

        #time
        T = VGroup(TexText("t = ", tex_to_color_map = {"t" : GREY}), DecimalNumber(0)).arrange()
        T.to_edge(LEFT).shift(UP*3)
        T.add_updater(lambda m : m[1].set_value(t.get_value()))

        #value tracker for R(t)
        Rt = ValueTracker(1.0)
        Rt.add_updater(lambda m : m.set_value(np.exp(-t.get_value())))

        Rvalue = VGroup(TexText("{R}({t}) = ", tex_to_color_map = {"{t}" : GREY, "{R}" : BLUE}), DecimalNumber(1)).arrange()
        Rvalue.next_to(T, DOWN, buff=0.3, aligned_edge=LEFT)
        Rvalue.add_updater(lambda m : m[1].set_value(Rt.get_value()))

        #dot which tracks R(t)
        Rtpos = Dot(color=BLUE)
        Rtpos.add_updater(lambda m : m.move_to(nl.n2p(Rt.get_value())))

        #line to track the value of R(t)
        RtLine = Line(nl.n2p(0), nl.n2p(1), stroke_width=6.0).set_color(BLUE)
        RtLine.add_updater(lambda m : m.put_start_and_end_on(nl.n2p(0), Rtpos.get_center()))

        #the arrow which tracks rdot
        rdot = Arrow(RIGHT, LEFT)
        rdot.set_color(YELLOW)
        def rdot_updater(m):
            l = get_norm(nl.n2p(Rt.get_value()) - nl.n2p(0))
            m.set_width(l)
            m.move_to(Rtpos.get_center() + l * LEFT * 0.5 + DOWN * 0.2)
        rdot.add_updater(rdot_updater)

        R = Tex("R").scale(0.6).set_color(BLUE)
        dR = Tex("d{R} \\over d{t}", tex_to_color_map = {"{R}" : BLUE, "{t}" : GREY}).scale(0.6)
        always(R.next_to, RtLine, UP)
        always(dR.next_to, rdot, DOWN)

        self.play(
            de2.animate.to_corner(UR)
        )
        self.play(
            ShowCreation(nl)
        )
        # self.add(nl)
        
        self.play(
            Write(T)
        )
        self.wait(2)
        self.play(
            GrowArrow(RtLine),
            run_time=2.0
        )
        self.play(
            GrowFromCenter(Rtpos)
        )
        self.play(
            Write(R),
            Write(Rvalue)
        )
        temp_arr = Arrow(LEFT, RIGHT).set_color(YELLOW).shift(UP)
        temp_arr.set_width(2)
        self.wait(2)
        self.play(
            GrowArrow(temp_arr)
        )
        
        temp_eq = Tex("{d{R}({0}) \\over d{t} } = -{R}({0})", tex_to_color_map = {"{R}" : BLUE, "{0}" : GREY})
        temp_eq.scale(0.7).shift(DOWN)
        self.wait(3)
        self.play(
            Write(temp_eq)
        )
        self.wait(2)
        self.play(
            temp_arr.animate.flip()
        )
        self.wait()
        
        self.play(
            temp_arr.animate.move_to(nl.n2p(0) + RIGHT + DOWN*0.2)
        )
        self.add(rdot)
        self.remove(temp_arr)
        self.wait(2)
        
        self.play(
            Write(dR)
        )
        self.play(
            FadeOut(temp_eq)
        )
        self.wait(3)
        self.add(t, Rt)
        self.wait(2)
        # self.play(
        #     FadeOut(dR),
        #     FadeOut(R),
        #     FadeOut(RtLine),
        #     FadeOut(rdot),
        #     FadeOut(Rtpos),
        #     FadeOut(T)
        # )

        self.play(
            FocusOn(nl.n2p(0))
        )
        self.wait(2)
        self.remove(t, Rt, T, Rvalue)
        
        carr = CurvedArrow(nl.n2p(0), nl.n2p(0) + DOWN*2, angle=TAU/4).set_color(GREEN)
        fp = TexText("Fixed Point").next_to(carr.get_end(), RIGHT)
        self.play(
            FocusOn(nl.n2p(0))
        )
        self.play(
            ShowCreation(carr)
        )
        self.play(
            Write(fp)
        )
        self.wait()

class RabbitFox(Scene):

    def construct(self):

        #the agentbased model
        pp = AgentBasedSim(box_size = 5).to_edge(LEFT)
        #separating rabbits and foxes
        rabbits, foxes = pp.get_cells_cooresponding_to_state(1), pp.get_cells_cooresponding_to_state(-1)

        self.play(
            ShowCreation(pp)
        )

        varR = Tex("{R}({t})", tex_to_color_map={"{R}" : BLUE, "{t}" : GREY})
        varF = Tex("{F}({t})", tex_to_color_map={"{F}" : RED, "{t}" : GREY})
        varR.shift(2*RIGHT + UP)
        varF.shift(2*RIGHT + DOWN)

        self.wait(3)
        self.play(
            TransformFromCopy(rabbits, varR)
        )
        self.wait(2)
        self.play(
            TransformFromCopy(foxes, varF)
        )
        # self.add(pp, varR, varF)
        to_isolate = ["(", ")", "d", "+", "-"]
        dRdt = Tex("{ d{R}({t}) \\over d{t} }", "=", "{k}{R}({t}) - {m}{R}({t}){F}({t})",
        tex_to_color_map={"{R}" : BLUE, "{t}" : GREY, "{F}" : RED},
        isolate = [*to_isolate]
        )
        dFdt = Tex("{ d{F}({t}) \\over d{t} }", "=", "-{n}{F}({t}) + {p}{R}({t}){F}({t})",
        tex_to_color_map={"{R}" : BLUE, "{t}" : GREY, "{F}" : RED},
        isolate = [*to_isolate]
        )
        dRdt.to_edge(UP).shift(RIGHT * 3)
        dFdt.next_to(dRdt, DOWN, buff=1.1, aligned_edge=LEFT)
        
        self.wait(3)
        self.play(
            TransformMatchingTex(varR, dRdt[:8])
        )
        self.wait()
        self.play(
            TransformMatchingTex(varF, dFdt[:8])
        )
        
        #isolating only rabbit and only fox cases
        self.wait(3)
        self.play(
            FlashAround(pp)
        )
        self.play(
            Write(dRdt[8 : 14])
        )
        self.wait(3)
        self.play(
            Write(dFdt[8 : 15])
        )
        self.wait(3)

        #interaction
        preys = rabbits.copy().arrange_in_grid(buff=0.05)
        preds = foxes.copy().arrange_in_grid(buff=0.05)
        preys.to_edge(DOWN).shift(RIGHT*1.5)
        preds.next_to(preys, RIGHT, buff=1.5)
        # self.add(preds, preys)
        self.play(
            TransformFromCopy(rabbits, preys)
        )
        self.wait(2)
        self.play(
            TransformFromCopy(foxes, preds)
        )
        self.wait(3)
        r1 = preys[0]
        self.play(
            r1.animate.scale(5)
        )
        interaction_line = Line(r1.get_center(), preds[0].get_center(), color=YELLOW, stroke_width=1.0)
        # interaction_line.add_tip()
        sq = Square(color=YELLOW).surround(preds[0]).scale(0.2)
        self.wait(2)
        self.play(
            GrowArrow(interaction_line)
        )
        self.play(
            ShowCreation(sq)
        )
        for i in range(1, len(preds)):
            self.play(
                interaction_line.animate.put_start_and_end_on(r1.get_center(), preds[i].get_center()),
                sq.animate.move_to(preds[i]),
                run_time=3.0/len(preds)
                )

        FtimesR = Tex("{R} \\cdot {F}", tex_to_color_map={"{R}" : BLUE, "{F}" : RED})
        FtimesR.scale(0.7).next_to(preds, UP, buff=0.6)

        self.wait(2)
        self.play(
            Write(FtimesR[-1])
        )

        self.play(
            FadeOut(interaction_line),
            r1.animate.scale(1/5.),
            FadeOut(sq)
        )

        self.wait(3)
        self.play(
            ApplyWave(preys)
        )

        self.wait()
        self.play(
            Write(FtimesR[:2])
        )

        total_interactions = TexText("Total Encounters").next_to(FtimesR, UP)

        self.wait(2)

        self.play(
            FlashAround(FtimesR)
        )
        self.play(
            Write(total_interactions)
        )
        self.wait(2)
        self.play(
            FadeOut(total_interactions)
        )
        self.wait(3)

        self.play(
            TransformFromCopy(FtimesR, dRdt[15 : ])
        )
        self.wait(2)
        self.play(
            FlashAround(dRdt[15])
        )
        self.play(
            FlashAround(dRdt[15])
        )

        self.wait(2)
        self.play(
            FlashAround(dRdt[14])
        )
        self.play(
            FlashAround(dRdt[14])
        )
        self.wait(2)
        self.play(
            Write(dRdt[14])
        )
        self.wait(3)

        self.play(
            TransformFromCopy(FtimesR, dFdt[16 : ])
        )
        self.wait()
        self.play(
            FlashAround(dFdt[16])
        )
        self.play(
            FlashAround(dFdt[16])
        )

        self.wait(2)
        self.play(
            FlashAround(dFdt[15])
        )
        self.play(
            FlashAround(dFdt[15])
        )
        self.wait(3)
        self.play(
            Write(dFdt[15])
        )
        self.wait()
        self.play(
            FadeOut(preds),
            FadeOut(preys),
            FadeOut(FtimesR)
        )

        pp.pauseorresume(True)

        self.wait(10)

class RabbitFoxSolution(Scene):

    def construct(self):
        axes = Axes(
            (0, 12),
            (0, 12),
            height=7,
            width=7
        )
        axes.to_edge(LEFT, buff=0.5)
        labelR = Tex("R").set_color(BLUE)
        labelF = Tex("F").set_color(RED)
        labelR.next_to(axes.x_axis.get_right(), UP)
        labelF.next_to(axes.y_axis.get_top(), RIGHT)
        self.play(
            ShowCreation(axes),
            Write(labelR),
            Write(labelF)
        )

        #equations
        to_isolate = ["(", ")", "+", "-", "=", "d", "t", "R", "F", "k", "m", "n", "p"]
        dRdt = Tex(
            "{dR(t) \\over dt} = k R(t) - mR(t)F(t)",
            isolate=[*to_isolate],
            tex_to_color_map = {"t" : GREY, "R" : BLUE, "F" : RED}
        ).scale(0.8)
        dFdt = Tex(
            "{dF(t) \\over dt} = -n F(t) + pR(t)F(t)",
            isolate=[*to_isolate],
            tex_to_color_map = {"t" : GREY, "R" : BLUE, "F" : RED}
        ).scale(0.8)

        self.wait(3)
        self.play(
            Write(dRdt),
        )
        self.play(
            dRdt.animate.to_corner(UR)
        )
        self.wait(2)
        self.play(
            Write(dFdt),
        )
        self.play(
            dFdt.animate.next_to(dRdt, DOWN)
        )

        #showing an example of a state
        stateDot = Dot(color=YELLOW)
        stateDot.move_to(axes.c2p(3, 5))
        Fline = Line(axes.c2p(3, 0), axes.c2p(3, 5),color=RED)
        Rline = Line(axes.c2p(0, 5), axes.c2p(3, 5),color=BLUE)

        state = DecimalMatrix([[3.0, 5.0]]).scale(0.7).next_to(stateDot, UR)
        

        def line_updater(m):
            c = stateDot.get_center()
            start = axes.c2p(0, axes.p2c(c)[1])
            m.put_start_and_end_on(start, c)
        Rline.add_updater(line_updater)
        def line_updater2(m):
            c = stateDot.get_center()
            start = axes.c2p(axes.p2c(c)[0], 0)
            m.put_start_and_end_on(start, c)
        Fline.add_updater(line_updater2)
        

        # self.add(Rline, Fline, stateDot, state)
        self.wait(3)
        self.play(
            GrowFromCenter(stateDot)
        )
        self.wait(2)
        self.play(
            GrowArrow(Rline),
            run_time=2
        )
        self.wait(2)
        self.play(
            GrowArrow(Fline),
            run_time=2
        )
        self.wait(2)

        rval = DecimalNumber(3.0, num_decimal_places=1)
        fval = DecimalNumber(5.0, num_decimal_places=1)
        rval.next_to(Rline, UP)
        fval.next_to(Fline, RIGHT)

        self.play(
            Write(rval)
        )
        self.wait()
        self.play(
            Write(fval)
        )

        self.wait()

        self.play(
            TransformFromCopy(VGroup(rval, fval), state)
        )

        def mat_updater(m):
            c = stateDot.get_center()
            x, y = axes.p2c(c)
            m.become(DecimalMatrix([[x, y]]))
        state.add_updater(mat_updater)
        self.add(state)
        always(state.next_to, stateDot, UR)

        self.play(
            FadeOut(rval),
            FadeOut(fval)
        )

        self.wait(2)
        self.play(
            stateDot.animate.move_to(axes.c2p(8, 7))
        )
        self.wait(2)
        self.play(
            stateDot.animate.move_to(axes.c2p(7, 2))
        )
        self.wait(2)
        self.play(
            stateDot.animate.move_to(axes.c2p(5, 6))
        )
        self.wait(2)
        self.play(
            FadeOut(Rline), 
            FadeOut(Fline)
            )

        #the velocity
        c = stateDot.get_center()
        horizontal_arr = Arrow(RIGHT, LEFT).set_color(GREEN)
        horizontal_arr.put_start_and_end_on(c, c + (DOWN + RIGHT*2)*0.6)
        self.wait(2)
        self.play(
            GrowArrow(horizontal_arr)
        )
        self.wait(2)
        self.play(
            horizontal_arr.animate.scale(2, about_point=horizontal_arr.get_start_and_end()[0])
        )
        self.wait()
        self.play(
            horizontal_arr.animate.scale(0.5, about_point=horizontal_arr.get_start_and_end()[0])
        )
        horizontal_arr.save_state()
        self.wait(2)
        self.play(
            UpdateFromAlphaFunc(horizontal_arr, lambda m, a: m.restore().rotate(TAU*a, about_point=horizontal_arr.get_start_and_end()[0]))
        )
        self.wait()
        self.play(
            UpdateFromAlphaFunc(horizontal_arr, lambda m, a: m.restore().rotate(-TAU*a, about_point=horizontal_arr.get_start_and_end()[0]))
        )

        #flash the horizontal direction
        self.wait(3)
        horizontal = VectorField(lambda x, y: np.array([2.0, 0.0, 0.0]), axes, step_multiple=1.0)
        self.play(
            ShowCreationThenFadeOut(horizontal),
            run_time=1
        )

        self.wait(2)
        self.play(
            dRdt.animate.move_to(ORIGIN + RIGHT + DOWN)
        )

        horizontal_vel = Tex(
            "= k \\cdot 5 - m \\cdot 5 \\cdot 6",
            isolate = ["k", "m", "5", "6", "-", "\\cdot"],
            tex_to_color_map = {"5" : BLUE, "6" : RED}
        ).scale(0.8)
        eqsym = dRdt.get_part_by_tex("=")
        horizontal_vel.next_to(eqsym, DOWN, aligned_edge=LEFT, buff=1.0)
        self.wait(2)
        self.play(
            Write(horizontal_vel[0]),
            Write(horizontal_vel[1])
        )
        self.wait(2)
        rr = Tex("R").set_color(BLUE)
        ff = Tex("F").set_color(RED)

        rr.next_to(state[0][0], UP)
        ff.next_to(rr, RIGHT, buff=1.0)

        self.play(
            FadeIn(rr)
        )
        self.play(
            FadeIn(ff)
        )

        self.wait(2)

        self.play(
            TransformFromCopy(state[0][0], horizontal_vel[3]),
            Write(horizontal_vel[2])
        )
        self.wait(2)
        self.play(
            Write(horizontal_vel[4]),
            Write(horizontal_vel[5]),
        )
        self.wait()
        self.play(
            Write(horizontal_vel[6]),
            TransformFromCopy(state[0][0], horizontal_vel[7])
        )
        self.play(
            Write(horizontal_vel[8]),
            TransformFromCopy(state[0][1], horizontal_vel[9])
        )
       
        self.wait(3)
        #k=1.0, m = 0.5
        k = horizontal_vel.get_part_by_tex("k")
        self.play(
            FlashAround(k)
        )
        self.wait()
        self.play(
            Transform(k, Tex("1").scale(0.8).move_to(k))
        )
        m = horizontal_vel.get_part_by_tex("m")
        self.wait(2)
        self.play(
            FlashAround(m)
        )
        self.wait()
        self.play(
            Transform(m, Tex("\\frac{1}{2}").scale(0.8).move_to(m))
        )

        h_result = Tex("=", "-10").scale(0.8)
        h_result.next_to(eqsym, DOWN, aligned_edge=LEFT, buff=1.0)
        self.wait(2)
        self.play(
            FadeTransform(horizontal_vel, h_result),
        )

        self.wait(3)

        #rotate the arrow
        self.play(
            horizontal_arr.animate.put_start_and_end_on(c, c + LEFT)
        )
        self.wait(2)
        self.play(
            horizontal_arr.animate.scale(10, about_point=c)
        )
        self.wait(3)
        self.play(
            horizontal_arr.animate.scale(2/10., about_point=c).set_color(BLUE)
        )
        
        #vertical
        self.wait(2)
        self.play(
            dRdt.animate.to_corner(UR),
            FadeOut(h_result),
            dFdt.animate.move_to(ORIGIN + RIGHT + DOWN)
        )

        #flash the horizontal direction
        self.wait(2)
        horizontal = VectorField(lambda x, y: np.array([2.0, 0.0, 0.0]), axes, step_multiple=1.0)
        self.play(
            ShowCreationThenFadeOut(horizontal.rotate(PI/2)),
            run_time=1
        )

        vertical_vel = Tex(
            "= - n \\cdot 6 + p \\cdot 5 \\cdot 6",
            isolate = ["n", "p", "5", "6", "-", "\\cdot"],
            tex_to_color_map = {"5" : BLUE, "6" : RED}
        ).scale(0.8)
        eqsym = dFdt.get_part_by_tex("=")
        vertical_vel.next_to(eqsym, DOWN, aligned_edge=LEFT, buff=1.0)
        self.wait(3)
        self.play(
            Write(vertical_vel[0]),
            Write(vertical_vel[1]),
            Write(vertical_vel[2])
        )
        self.wait(2)
        self.play(
            TransformFromCopy(state[0][1], vertical_vel[4]),
            Write(vertical_vel[3])
        )
        self.wait(2)
        self.play(
            Write(vertical_vel[5]),
            Write(vertical_vel[6]),
        )
        self.wait(2)
        self.play(
            Write(vertical_vel[7]),
            TransformFromCopy(state[0][0], vertical_vel[8])
        )
        self.wait()
        self.play(
            Write(vertical_vel[9]),
            TransformFromCopy(state[0][1], vertical_vel[10])
        )
       
        self.wait(3)
        #n=1.5, p = 0.5
        n = vertical_vel.get_part_by_tex("n")
        self.play(
            FlashAround(n)
        )
        self.wait()
        self.play(
            Transform(n, Tex("\\frac{3}{2}").scale(0.8).move_to(n))
        )
        p = vertical_vel.get_part_by_tex("p")
        self.wait(3)
        self.play(
            FlashAround(p)
        )
        self.play(
            Transform(p, Tex("\\frac{1}{2}").scale(0.8).move_to(p))
        )
        self.wait(3)

        v_result = Tex("=", "6").scale(0.8)
        v_result.next_to(eqsym, DOWN, aligned_edge=LEFT, buff=1.0)
        self.play(
            FadeTransform(vertical_vel, v_result),
        )

        self.wait(2)

        vertical_arr = Arrow(LEFT, RIGHT).set_color(RED)
        vertical_arr.put_start_and_end_on(c, c + UP * 6/5.0)
        self.play(
            GrowArrow(vertical_arr)
        )

        self.wait(2)
        self.play(
            dFdt.animate.next_to(dRdt, DOWN, aligned_edge=LEFT),
            FadeOut(v_result),
        )

        #cut to the vector addition scene here!!

        #add vectors
        self.wait(2)
        self.play(
            horizontal_arr.animate.shift(UP * 6/5.0)
        )
        total_arr = Arrow().set_color(GOLD)
        total_arr.put_start_and_end_on(c, horizontal_arr.get_end())

        self.play(
            GrowArrow(total_arr)
        )
        state.clear_updaters()
        self.wait(2)
        self.play(
            FadeOut(horizontal_arr),
            FadeOut(vertical_arr),
            FadeOut(state),
            FadeOut(rr),
            FadeOut(ff)
        )

        vector_field = VectorField(get_vecfield_preypred(1.0, 0.5, 1.5, 0.5), axes)
        self.wait(2)
        self.play(
            total_arr.animate.become(vector_field.get_vector(axes.p2c(c)))
        )
        self.wait(2)

        def vel_updater(m):
            c = stateDot.get_center()
            m.become(vector_field.get_vector(axes.p2c(c)))

        total_arr.add_updater(vel_updater)
        self.add(total_arr)

        def along_vector_field(m, dt):
            p = m.get_center()
            c = axes.p2c(p)
            v = get_vecfield_preypred(1.0, 0.5, 1.5, 0.5)(c[0], c[1])
            m.shift(v * dt)
        stateDot.add_updater(along_vector_field)

        trail = Trail(stateDot, color=interpolate_color(BLUE, RED, 0.5))
        self.add(trail)
        self.wait(10)

        total_arr.clear_updaters()
        stateDot.clear_updaters()
        self.remove(trail)
        self.play(
            FadeOut(stateDot),
            FadeOut(total_arr)
        )
        self.wait(3)

        self.play(
            ShowCreation(vector_field)
        )
        self.wait(3)
        
        #stream lines
        stream_lines = StreamLines(
            get_vecfield_preypred(1.0, 0.5, 1.5, 0.5), axes,
            magnitude_range=(0, 12),
            stroke_width=4,
            step_multiple=0.6,
            n_samples_per_line=20
        )


        self.play(
            FadeOut(vector_field)
        )
        animated_stream_lines = AnimatedStreamLines(stream_lines)
        self.add(animated_stream_lines)
        self.wait(10)

        frame = self.camera.frame
        frame.save_state()

        #zoom1
        rect1 = ScreenRectangle().scale(0.8).move_to(axes.c2p(0, 0))
        self.play(
            frame.animate.replace(rect1)
        )
        self.wait(10)
        self.play(
            frame.restore
        )

        self.wait(2)

        saddle = TexText("Saddle Point").shift(RIGHT * 4)
        self.play(
            Write(saddle)
        )
        self.wait()
        self.play(
            FadeOut(saddle)
        )

        self.wait(3)

        self.play(
            FlashAround(dFdt)
        )
        self.play(
            FlashAround(dRdt)
        )

        fp = Tex("\\left[ \\frac{n}{p}, \\frac{k}{m} \\right]")
        fp.shift(RIGHT * 4)

        self.play(
            Write(fp)
        )
        self.wait(3)

        fp2 = Tex("\\left[ \\frac{1.5}{0.5}, \\frac{1.0}{0.5} \\right]").shift(RIGHT * 4)
        self.play(
            ReplacementTransform(fp, fp2)
        )

        fp3 = Tex("\\left[ 3.0, 2.0 \\right]").shift(RIGHT * 4)
        self.wait(3)

        self.play(
            ReplacementTransform(fp2, fp3)
        )

        #zoom2
        self.wait(2)
        rect2 = ScreenRectangle().scale(0.7).move_to(axes.c2p(3, 2))
        self.play(
            frame.animate.replace(rect2),
            FadeOut(fp3)
        )
        self.wait(8)
        self.play(
            frame.restore
        )
        
        self.wait(2)
        center = TexText("Center").shift(RIGHT * 4)
        self.play(
            Write(center)
        )
        self.wait()
        self.play(
            FadeOut(center)
        )
        self.wait(3)

#highly inefficient!! should reuse the AgentBasedSimGraph here!!
class GraphOfState(VGroup):
    CONFIG = {
        "update_freq" : 1/60.0
    }

    def __init__(self, point, parent_axes, **kwargs):
        super().__init__(**kwargs)
        self.point = point
        self.parent_axes = parent_axes
        self.time = 0.0
        self.last_update_time = -np.inf
        self.isUpdating = True
        self.initialise_counts()
        self.setup_axes()
        self.setup_graph()
        self.add_ticks()
        self.add_legends()

        self.add_updater(lambda m, dt: m.update_time(dt))
        self.add_updater(lambda m, dt : m.update_graphs(dt))
        self.add_updater(lambda m, dt : m.update_ticks(dt))

    def initialise_counts(self):
        c1, c2 = self.parent_axes.p2c(self.point.get_center())
        self.counts = [np.array([c1, c2])/15]

    def setup_axes(self):
        axes = Axes((0, 1.0), (0.0, 1.1, 0.2), height=5, width=5, y_axis_config = {
                "tick_frequency" : 0.1
            })
        axes.add_coordinate_labels(x_values= [], y_values=np.arange(0.0, 1.1, 0.2), num_decimal_places=1, font_size=24)
        self.axes = axes
        self.add(self.axes)

    def add_ticks(self):
        self.x_ticks = VGroup()
        self.x_labels = VGroup()
        self.add(self.x_ticks, self.x_labels)

    def add_legends(self):
        pred = VGroup(Line(LEFT, RIGHT).set_width(0.3).set_color(RED), Text("Fox").scale(0.4)).arrange(buff=0.1)
        prey = VGroup(Line(LEFT, RIGHT).set_width(0.3).set_color(BLUE), Text("Rabbit").scale(0.4)).arrange(buff=0.1)
        pred.move_to(self.axes.c2p(0.8, 1))
        prey.next_to(pred, DOWN, aligned_edge=LEFT)
        self.add(pred, prey)

    def setup_graph(self):
        self.graph = self.get_graph()
        self.add(self.graph)

    def update_time(self, dt):
        if self.isUpdating:
            self.time += dt

    def get_graph(self):
        axes = self.axes
        counts = self.counts

        preds = []
        preys = []
        for x, counts in zip(np.linspace(0.0, 1.0, len(counts)), counts):
            prey = axes.c2p(x, counts[0])
            pred = axes.c2p(x, counts[1])
            preds.append(pred)
            preys.append(prey)

        prey_line = VGroup()
        for i in range(len(preys)-1):
            prey_line.add(Line(preys[i], preys[i+1], color=BLUE, stroke_width=4.0))

        pred_line = VGroup()
        for i in range(len(preds)-1):
            pred_line.add(Line(preds[i], preds[i+1], color=RED, stroke_width=4.0))

        region = VGroup(pred_line, prey_line)
        return region

    def update_ticks(self, dt):
        if self.isUpdating:

            tick_height = 0.03 * self.get_height()
            tick_template = Line(DOWN, UP).set_height(tick_height)

            if 0 < self.time < 10:
                tick_range = range(0, int(self.time)+1, 1)
            elif self.time < 50:
                tick_range = range(5, int(self.time)+1, 5)
            elif self.time < 100:
                tick_range = range(10, int(self.time)+1, 10)
            else:
                tick_range = range(20, int(self.time)+1, 20)

            def get_tick(x):
                tick = tick_template.copy()
                tick.move_to(self.axes.c2p(x/self.time, 0))
                return tick

            def get_label(x, tick):
                label = Integer(x)
                label.set_height(tick_height)
                label.next_to(tick, DOWN, buff=0.2*tick_height)
                return label

            x_ticks = VGroup()
            x_labels = VGroup()
            for x in tick_range:
                tick = get_tick(x)
                x_ticks.add(tick)
                x_labels.add(get_label(x, tick))

            self.x_ticks.become(x_ticks)
            self.x_labels.become(x_labels)

    def get_data(self):
        c1, c2 = self.parent_axes.p2c(self.point.get_center())
        return np.array([c1, c2])/15

    def update_graphs(self, dt):
        if self.isUpdating:

            if (self.time - self.last_update_time) > self.update_freq:
                self.last_update_time = self.time
                self.counts.append(self.get_data())
                self.graph.become(self.get_graph())

class Trail(VMobject):
    CONFIG = {
        "color" : YELLOW,
    }

    def __init__(self, mobject, path_length=250, **kwargs):
        super().__init__(**kwargs)
        self.parent = mobject
        self.path_length = path_length
        self.start_new_path(self.parent.get_center())
        self.add_updater(lambda m : m.update_lines())

    def update_lines(self):
        if self.get_num_points() > self.path_length:
            self.reverse_points()
            #this will resize the points array by chopping the end. the reverse in front of it will make the initial points go!
            self.resize_points(self.path_length - 1)
            self.reverse_points()
        self.add_smooth_curve_to(self.parent.get_center())
        self.make_approximately_smooth()
        
class CompareToData(Scene):

    def construct(self):
        axes = Axes(
            (0, 12),
            (0, 12),
            height=7,
            width=7
        )
        axes.to_edge(LEFT, buff=0.5)
        labelR = Tex("R").set_color(BLUE)
        labelF = Tex("F").set_color(RED)
        labelR.next_to(axes.x_axis.get_right(), UP)
        labelF.next_to(axes.y_axis.get_top(), RIGHT)
        self.play(
            ShowCreation(axes),
            Write(labelR),
            Write(labelF)
        ) 

        stateDot = Dot(color=YELLOW)
        stateDot.move_to(axes.c2p(3, 5))

        self.play(
            GrowFromCenter(stateDot)
        )
        self.wait(2)

        Fline = Line(axes.c2p(3, 0), axes.c2p(3, 5)).set_color(RED)
        Rline = Line(axes.c2p(0, 5), axes.c2p(3, 5)).set_color(BLUE)
        Rval = DecimalNumber(3.0, num_decimal_places=1)
        always(Rval.next_to, Rline, UP)
        Fval = DecimalNumber(5.0, num_decimal_places=1)
        always(Fval.next_to, Fline, RIGHT)

        self.play(
            GrowArrow(Rline),
            run_time=1.5
        )
        self.play(
            Write(Rval)
        )
        self.wait(2)
        self.play(
            GrowArrow(Fline),
            run_time=1.5
        )
        self.play(
            Write(Fval)
        )

        vector_field = VectorField(get_vecfield_preypred(1.0, 0.5, 1.5, 0.5), axes)
        def along_vector_field(m, dt):
            p = m.get_center()
            c = axes.p2c(p)
            v = get_vecfield_preypred(1.0, 0.5, 1.5, 0.5)(c[0], c[1])
            m.shift(v * dt)
        
        data_axes = GraphOfState(stateDot, axes).to_edge(RIGHT)
        data_axes.isUpdating = False

        self.play(
            ShowCreation(data_axes)
        )
        self.wait(3)
        stateDot.add_updater(along_vector_field)
        path = Trail(stateDot, color=interpolate_color(BLUE, RED, 0.5))
        data_axes.isUpdating = True
        
        lines = VGroup(Rline, Fline)
        vals = VGroup(Rval, Fval)

        def update_lines(vg):
            p = stateDot.get_center()
            x, y = axes.p2c(p)
            vg[0].put_start_and_end_on(axes.c2p(0, y), p)
            vg[1].put_start_and_end_on(axes.c2p(x, 0), p)

        def update_vals(vg):
            p = stateDot.get_center()
            x, y = axes.p2c(p)
            vg[0].set_value(x)
            vg[1].set_value(y)


        lines.add_updater(update_lines)
        vals.add_updater(update_vals)
        self.add(stateDot, path, lines, vals)
        self.wait(20)
        data_axes.isUpdating = False
        vals.clear_updaters()
        lines.clear_updaters()
        stateDot.clear_updaters()
        self.remove(path)
        self.play(
            FadeOut(axes),
            FadeOut(labelF),
            FadeOut(labelR),
            FadeOut(stateDot),
            FadeOut(vals),
            FadeOut(lines)
        )

        self.wait(3)
        realgraph = ImageMobject(
            "graph.jpg"
        )
        realgraph.next_to(data_axes, LEFT, aligned_edge=DOWN)
        source = TexText("source : wikipedia").next_to(realgraph, DOWN, aligned_edge=LEFT)
        self.play(
            FadeInFromPoint(realgraph, realgraph.get_bottom())
        )
        self.play(
            Write(source)
        )
        self.wait(2)
        params = Tex("parameters = \\{k, \\, m, \\, n, \\, p \\}").next_to(realgraph, UP, buff=1.0)
        self.play(
            Write(params)
        )
        self.wait()
        self.play(
            ApplyWave(params)
        )
        self.wait(5)

class LetsTalkLove(Scene):

    def construct(self):
        f = Firework(shape=heart)
        self.add(f)
        self.wait(4)
        f.stop()
        self.wait(2)
        self.remove(f)
        to_isolate = ["(", ")", "d", "t", "B", "G", "a", "b", "=", "-", "+"]
        dBdt = Tex(
            "{dB(t) \\over dt} = a G(t)",
            isolate=[*to_isolate],
            tex_to_color_map = {"B" : RED, "G": GREEN, "t":GREY}
        )
        dGdt = Tex(
            "{dG(t) \\over dt} = - b B(t)",
            isolate=[*to_isolate],
            tex_to_color_map = {"B" : RED, "G": GREEN, "t":GREY}
        )

        eqns = VGroup(dBdt, dGdt).arrange_in_grid(2, 1)

        self.wait()
        self.play(
            Write(dBdt[:8])
        )
        
        self.play(
            Write(dGdt[:8])
        )
        self.wait(2)
        self.play(
            FlashAround(dBdt[1])
        )
        btotext = CurvedArrow(dBdt[1].get_top(), dBdt[1].get_top() + RIGHT * 2, angle=-TAU/4).set_color(YELLOW)
        boyslove = Text("Boy's love for the Girl",
        font_size=18,
        t2c = {"Boy" : RED, "Girl" : GREEN})
        boyslove.next_to(btotext.get_end(), DOWN).shift(boyslove.get_width() / 2 * RIGHT)
        self.wait()
        self.play(
            ShowCreation(btotext)
        )
        self.play(
            Write(boyslove)
        )

        self.wait(2)
        self.play(
            FlashAround(dGdt[1])
        )
        gtotext = CurvedArrow(dGdt[1].get_top(), dGdt[1].get_top() + LEFT * 2, angle=TAU/4).set_color(YELLOW)
        girlslove = Text("Girl's love for the Boy",
        font_size=18,
        t2c = {"Boy" : RED, "Girl" : GREEN})
        girlslove.next_to(gtotext.get_end(), DOWN).shift(girlslove.get_width() / 2 * LEFT)
        self.wait()
        self.play(
            ShowCreation(gtotext)
        )
        self.play(
            Write(girlslove)
        )

        self.wait(2)
        self.play(
            FadeOut(btotext),
            FadeOut(boyslove),
            FadeOut(gtotext),
            FadeOut(girlslove)
        )
        self.wait(3)
        self.play(
            Write(dBdt[8:])
        )

        self.wait(2)
        rhsofB = Text(
            "Boy tends to love the Girl more when she loves back",
            font_size=18,
            t2c = {"Boy" : RED, "Girl" : GREEN}
        )
        rhsofB.next_to(dBdt, UP)
        self.play(
            Write(rhsofB)
        )
        self.wait(3)

        
        self.play(
            Write(dGdt[8:])
        )
        self.wait(2)
        rhsofG = Text(
            "Girl tends to love the Boy more when he is rude to her!!!",
            font_size=18,
            t2c = {"Boy" : RED, "Girl" : GREEN}
        )
        rhsofG.next_to(dGdt, DOWN)
        self.play(
            Write(rhsofG)
        )

        self.wait(3)
        self.play(
            FadeOut(rhsofB),
            FadeOut(rhsofG)
        )
        self.play(
            eqns.animate.scale(0.6).to_corner(UR)
        )

        np = NumberPlane(
            background_line_style= {
            "stroke_color": BLUE_D,
            "stroke_width": 1,
            "stroke_opacity": 0.5,
        }
        )

        self.play(
            ShowCreation(np)
        )
        
        stream_lines = StreamLines(
            get_vecfield_toxiclove(1, 1),
            np,
            magnitude_range=(0, 8),
            stroke_width=4,
            step_multiple=0.3
        )
        animated_lines = AnimatedStreamLines(stream_lines)
        self.wait(3)
        self.add(animated_lines)
        self.wait(12)

        vec_field = VectorField(
            get_vecfield_toxiclove(1, 1),
            np
        )

        stateDot = Dot(color=YELLOW).shift(UP * 2 + RIGHT * 0.5)

        hislove = Arrow(ORIGIN, UP).set_color(RED)
        herlove = Arrow(ORIGIN, UP).set_color(GREEN)
        hislove.to_edge(LEFT)
        herlove.to_edge(RIGHT)
        love = VGroup(hislove, herlove)

        
        self.play(
            FadeOut(animated_lines)
        )
        self.wait(2)
        self.play(
            GrowFromCenter(stateDot)
        )
        self.wait()
        boy = TexText("Boy").set_color(RED).next_to(hislove, RIGHT, aligned_edge = DOWN)
        self.play(
            GrowArrow(hislove),
            Write(boy)
        )
        self.wait()
        girl = TexText("Girl").set_color(GREEN).next_to(herlove, LEFT, aligned_edge = DOWN)
        self.play(
            GrowArrow(herlove),
            Write(girl)
        )

        self.wait(3)

        def along_vector_field(m, dt):
            p = m.get_center()
            c = np.p2c(p)
            v = get_vecfield_toxiclove(1.0, 1.0)(c[0], c[1])
            m.shift(v * dt)
        stateDot.add_updater(along_vector_field)
        trail = Trail(stateDot, color=GOLD)


        def love_update(mob):
            c = stateDot.get_center()
            b, g = np.p2c(c)
            start0, end0 = mob[0].get_start_and_end()
            mob[0].put_start_and_end_on(start0, start0 + UP * b)
            start1, end1 = mob[1].get_start_and_end()
            mob[1].put_start_and_end_on(start1, start1  + UP * g)

        love.add_updater(love_update)

        self.add(stateDot, trail, love)

        self.wait(15)

        stateDot.clear_updaters()
        love.clear_updaters()
        self.remove(trail)
        self.play(
            FadeOut(stateDot),
            FadeOut(love),
            FadeOut(boy),
            FadeOut(girl)
        )

        self.play(
            eqns.animate.scale(1/0.6).move_to(ORIGIN)
        )

        new_dGdt = Tex(
            "{dG(t) \\over dt} = - b B(t) - c G(t)",
            isolate=[*to_isolate, "c"],
            tex_to_color_map = {"B" : RED, "G": GREEN, "t":GREY}
        )
        new_dGdt.move_to(dGdt)

        self.play(
            TransformMatchingTex(dGdt, new_dGdt)
        )

        self.wait(2)
        brace = Brace(new_dGdt[15:], DOWN)
        newtermis = Text(
            """
            She feels scared when her love 

            for the rude boyfriend is too much!!
            """,
            font_size=24,
        ).next_to(brace, DOWN)

        self.play(
            Write(newtermis),
            ShowCreation(brace)
        )
        self.wait(4)
        self.play(
            FadeOut(brace),
            FadeOut(newtermis)
        )

        self.play(
            VGroup(dBdt, new_dGdt).animate.scale(0.6).to_corner(UR)
        )

        new_streamlines = StreamLines(
            get_vecfield_toxicbutcarefullove(1, 1, 1),
            np,
            magnitude_range=(0, 8),
            stroke_width=4,
            step_multiple=0.3
        )
        new_animatedlines = AnimatedStreamLines(new_streamlines)
        self.add(new_animatedlines)
        self.wait(10)

class LorenzAttractor(Scene):
    CONFIG = {
        "camera_class" : ThreeDCamera,
    }

    def construct(self):
        
        axes = ThreeDAxes()
        vector_field = VectorField(
            get_vecfield_lorenzattractor(10.0, 8./3, 28.),
            axes
        )

        states = []

        for _ in range(30):
            states.append(Dot(color=random.choice(COLORMAP_3B1B)).move_to(axes.c2p(random.random()*20 - 10, 0.0, 0.0)))
        # state = Dot(color=YELLOW).move_to(axes.c2p(0.01, 0.0, 0.0))
        def along_vector_field(m, dt):
            p = m.get_center()
            c = axes.p2c(p)
            v = get_vecfield_lorenzattractor(10.0, 8./3, 28.)(c[0], c[1], c[2])
            m.shift(v * dt)

        frame = self.camera.frame
        frame.set_width(120)
        # frame.shift(UP * 100)
        # frame.set_phi(PI/2)
        # frame.set_gamma(PI)
        
        
        for state in states:
            state.add_updater(along_vector_field)

        trails = []
        for state in states:
            trails.append(Trail(state, 499, color=state.get_color()))
        
        self.add(*states, *trails)

        self.wait(25)

class Modelling(Scene):

    def construct(self):
        
        title = VGroup(
            TexText("Modelling").set_color(GOLD),
            Underline(TexText("Modelling")).set_color(GOLD)
        )
        title.to_edge(UP)
        self.play(
            Write(title)
        )

        self.wait(2)
        #the first thing that springs to mind is!
        fashionModel = ImageMobject("fashionModel.jpg")

        self.play(
            FadeIn(fashionModel)
        )
        self.wait(3)

        self.play(
            fashionModel.animate.shift(RIGHT * 6)
        )

        #what we are going to discuss
        meme = ImageMobject("drake_meme.jpg")
        meme.set_height(FRAME_HEIGHT - 2)
        meme.stretch_to_fit_width(9)
        meme.shift(DOWN)
        # self.add(meme)
        

        #animate
        def func(m):
            m.set_width(meme.get_width()/2, stretch=True)
            m.set_height(meme.get_height() / 2 - 0.2, stretch=True)
            m.move_to(meme.get_center() + meme.get_width()/4 * RIGHT + (meme.get_height()/4 + 0.1) * UP)
            return m
        # fashionModel.set_width(meme.get_width()/2, stretch=True)
        # fashionModel.set_height(meme.get_height() / 2, stretch=True)
        # fashionModel.move_to(meme.get_center() + meme.get_width()/4 * RIGHT + meme.get_height()/4 * UP)
        # self.add(fashionModel)
        self.play(
            FadeIn(meme),
        )
        self.bring_to_front(fashionModel)
        self.play(
            ApplyFunction(func, fashionModel)
        )
        self.wait()

        #scientific model
        sciModel = ImageMobject("sciModel.jpg")
        sciModel.set_width(meme.get_width()/2, stretch=True)
        sciModel.set_height(meme.get_height() / 2 + 0.2, stretch=True)
        sciModel.move_to(meme.get_center() + meme.get_width()/4 * RIGHT + (meme.get_height()/4 - 0.1) * DOWN)
        self.play(
            FadeIn(sciModel)
        )
        self.wait(2)

        self.play(
            FadeOut(fashionModel),
            FadeOut(meme),
            FadeOut(sciModel)
        )

        new_title = VGroup(
            TexText("Scientific \\, Modelling").set_color(ORANGE),
            Underline(TexText("Scientific \\, Modelling")).set_color(ORANGE)
        )
        new_title.to_edge(UP)

        self.play(
            TransformMatchingShapes(title, new_title)
        )
        self.wait(2)

        defofscimodels = Text(
            """
            They are the representations of real phenomena using

            physical or abstract objects. 
            """,
            font_size=24
        ).set_color("#f1faee")
        self.play(
            ShowIncreasingSubsets(defofscimodels)
        )
        self.wait()
        self.play(
            defofscimodels.animate.shift(UP * 2)
        )

        modelsareapprox = Text(
            """
            Models are at best approximations 
            
            of the real phenomena. 
            """,
            font_size=24
        ).set_color("#48cae4")

        self.play(
            Write(modelsareapprox)
        )
        self.wait()
        self.play(
            modelsareapprox.animate.next_to(defofscimodels, DOWN, aligned_edge=LEFT)
        )

        selfupdating = Text(
            """
            Hence scientists have to keep on 
            
            updating their models to get a 
            
            better and better description of reality. 
            """,
            font_size=24
        ).shift(DOWN).set_color("#a8dadc")
        self.play(
            ShowIncreasingSubsets(selfupdating)
        )
        self.wait(2)

        self.play(
            FadeOut(defofscimodels),
            FadeOut(modelsareapprox),
            FadeOut(selfupdating)
        )

        self.wait()

        quote = Text(
            """
            "...they(Science) mainly make models. 
            
            By a model is meant a mathematical construct which, 
            
            with the addition of certain verbal interpretations, 
            
            describes observed phenomena. 
            
            The justification of such a mathematical construct 
            
            is solely and precisely that it is expected to work—
            
            that is, correctly to describe phenomena 
            
            from a reasonably wide area"
            
                                            -- John von Neumann
            """,

            font_size=20,
            t2c = {
                "John von Neumann" : YELLOW,
            },
            t2s = {
                "model" : ITALIC
            }
        ).to_edge(RIGHT)

        Neumann = ImageMobject("vonNeumann.jpg").to_edge(LEFT)
        self.play(
            FadeIn(Neumann)
        )
        self.play(
            ShowIncreasingSubsets(quote),
            run_time=4.0
            )

        self.wait(5)

        self.play(
            quote.get_part_by_text("describes observed phenomena").animate.set_color(GREEN)
        )
        self.play(
            FlashAround(quote.get_part_by_text("describes observed phenomena"), color=GREEN),
            run_time=2.0
        )

        self.wait(3)
        self.play(
            quote.get_part_by_text("it is expected to work").animate.set_color(BLUE)
        )
        self.play(
            FlashAround(quote.get_part_by_text("it is expected to work"), color=BLUE),
            run_time=2.0
        )
        # self.add(sciModel)

class Types(Scene):

    def construct(self):
        title = VGroup(
            TexText("Scientific \\, Modelling").set_color(ORANGE),
            Underline(TexText("Scientific \\, Modelling")).set_color(ORANGE)
        )
        title.to_edge(UP)

        self.play(
            Write(title)
        )
        self.wait(2)

        types = ImageMobject("typesofmodels.jpg")
        types.set_width(FRAME_WIDTH - 1)
        types.set_height(FRAME_HEIGHT-3, stretch=True)

        rect = Rectangle(width=2, height=0.25, color=YELLOW, fill_opacity=0.3, stroke_width=0.0)
        rect.move_to(types.get_bottom() + UP * 0.45 + LEFT * 2.4)
        line = Line(rect.get_corner(DR) + LEFT*0.17, rect.get_corner(DL)+RIGHT*0.2, color="#d00000", stroke_width=8)
        line.shift(DOWN * 0.05)
        
        self.play(
            FadeIn(types)
        )
        self.wait(3)
        self.play(
            ShowCreation(line)
        )
        # self.play(
        #     ShowCreation(rect)
        # )
        # self.play(
        #     FlashAround(rect, color=RED)
        # )
        # self.play(
        #     FlashAround(rect, color=RED)
        # )
        self.wait(2)

        # self.add(types, rect)

class DynamicalSystems(Scene):

    def construct(self):
        title = VGroup(
            TexText("Dynamical \\, Systems").set_color(ORANGE),
            Underline(TexText("Dynamical \\, Systems")).set_color(ORANGE)
        )
        title.to_edge(UP)

        self.play(
            Write(title)
        )
        self.wait(2)

        definiton = Text(
            """
            A system which changes with time!! 
            """,
            font_size=36
        ).set_color("#fcbf49")

        self.play(
            Write(definiton)
        )
        self.wait(2)

        self.play(
            FadeOut(definiton)
        )

        clock = Clock()
        clock.to_corner(UL)

        self.play(
            ShowCreation(clock)
        )

        ball = Ball().to_corner(DL)
        balltrail = Trail(ball, 49, color=BLUE)

        self.add(ball, balltrail)

        clockanim = ClockPassesTime(clock, run_time=1, hours_passed=1)
        turn_animation_into_updater(clockanim, cycle=True)
        self.wait(3)
        self.remove(ball, balltrail)
        
        #pendulum fudge
        string = Line(UP * 2, DOWN, stroke_width=8.0, color=GREY)
        bob = Circle(radius=0.25, color=GREEN, fill_opacity=1.0)
        pivot, hang = string.get_start_and_end()
        def bob_pos(m):
            start, end = string.get_start_and_end()
            m.move_to(end)

        bob.add_updater(bob_pos)
        self.add(string, bob)
        self.play(
            string.animate.rotate(PI/4, about_point=pivot)
        )
        
        def get_swing(angle):
            def swing(m, alpha):
                m.restore()
                m.rotate(angle * alpha, about_point=pivot)
            return swing

        def swing_once():
            string.save_state()
            self.play(
                UpdateFromAlphaFunc(string, get_swing(-PI/2))
            )
            string.save_state()
            self.play(
                UpdateFromAlphaFunc(string, get_swing(PI/2))
            )
        bobtrail = Trail(bob, 49, color=GREEN)
        self.add(bobtrail)
        swing_once()
        swing_once()
        # swing_once()

        # self.play(
        #     ClockPassesTime(clock, run_time=5, hours_passed=24)
        # )

#simple ball with gravity
class Ball(VGroup):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.vel = np.array(
            [7.0, 10.0, 0.0]
        )

        self.body = Circle(color=BLUE, fill_opacity=1.0, radius=0.2)
        self.add(self.body)
        self.add_updater(lambda m, dt : m.fall(dt))

    def fall(self, dt):
        g = -9.81 * UP
        self.vel += g * dt
        self.shift(self.vel * dt)

class FeynmanQuote(Scene):

    def construct(self):
        
        sofar = TexText("So far so good!!").scale(1.5)
        self.play(
            Write(sofar)
        )
        self.wait(3)

        self.play(
            FadeOut(sofar)
        )



        quote = Text(
            """
            "If it disagrees with experiment,
            
             it is wrong!!"
            
                        -- Richard Feynman
            """,

            font_size=30,
            t2c = {
                "Richard Feynman" : YELLOW,
                "disagrees" : RED,
                "wrong" : RED
            }
        ).to_edge(RIGHT)

        Neumann = ImageMobject("feynman.jpg").to_edge(LEFT)
        self.play(
            FadeIn(Neumann)
        )
        self.play(
            ShowIncreasingSubsets(quote),
            run_time=4.0
            )

class Assumptions(Scene):

    def construct(self):

        title = VGroup(
            TexText("Assumptions").set_color(YELLOW),
            Underline(TexText("Assumptions")).set_color(YELLOW)
        )
        title.to_edge(UP)

        self.play(
            Write(title)
        )
        self.wait(2)

        pt1 = VGroup(
            Dot(),
            TexText("Rabbits do not die naturally!")
        ).arrange()

        pt2 = VGroup(
            Dot(),
            TexText("Infinite grass!")
        ).arrange()

        pt3 = VGroup(
            Dot(),
            TexText("Environment doesn't change")
        ).arrange()

        pt4 = VGroup(
            Dot(),
            TexText("Foxes doesn't have any other food source")
        ).arrange()

        points = VGroup(pt1, pt2, pt3, pt4).arrange_in_grid(4, 1, aligned_edge=LEFT, buff=1.0)
        points.to_edge(LEFT)
        points.fade(0.9)
        pt1.fade(0.0)
        self.play(
            FadeIn(points)
        )

        self.wait(3)

        self.play(
            pt1.animate.fade(0.9),
            pt2.animate.fade(0.0)
        )

        self.wait(3)

        self.play(
            pt2.animate.fade(0.9),
            pt3.animate.fade(0.0)
        )

        self.wait(3)

        self.play(
            pt3.animate.fade(0.9),
            pt4.animate.fade(0.0)
        )

        self.wait(3)

        self.play(
            pt4.animate.fade(0.9),
            pt1.animate.fade(0.0)
        )

        #updated eqns
        to_isolate = ["R", "F", "(", ")", "-", "=", "t", "\\over", "\\left(", "\\left[", "\\right)", "\\right]", "K_1", "+", 
        "r_1", "1", "a_{12}", "a_{21}", "K_2", "r_2"]
        dRdt = Tex(
            "{dR(t) \\over dt} = r_1 R(t) \\left[1 - \\left({R(t) + a_{12}F(t) \\over K_1} \\right)\\right]",
            isolate = to_isolate,
            tex_to_color_map = {"F" : RED, "R" : BLUE, "t" : GREY}
        )
        dFdt = Tex(
            "{dF(t) \\over dt} = r_2 F(t) \\left[1 - \\left({F(t) + a_{21}R(t) \\over K_2} \\right)\\right]",
            isolate = to_isolate,
            tex_to_color_map = {"F" : RED, "R" : BLUE, "t" : GREY}
        )

        eqns = VGroup(
            dRdt,
            dFdt
        ).arrange_in_grid(2, 1, aligned_edge=LEFT)
        eqns.scale(0.6).to_edge(RIGHT)
        
        self.play(
            FadeIn(eqns)
        )
        self.wait()
        # self.add(points)

class WhoAmI(Scene):

    def construct(self):
        vidicon = ImageMobject("video_icon.jpg").scale(0.7)
        vidicon.shift(LEFT * 3.5)
        sciencesort = Text("science.sort()")
        sciencesort.next_to(vidicon)
        hello = TexText("Hello!!").scale(2)
        self.play(
            FadeInFromPoint(hello, hello.get_bottom() + DOWN)
        )
        self.wait(2)
        self.play(
            FadeOutToPoint(hello, hello.get_top() + UP)
        )
        # self.play(
        #     FadeIn(vidicon),
        # )
        # self.play(
        #     Write(sciencesort)
        # )
        # self.add(vidicon, sciencesort)
        # self.wait(2)

class TakeHome(Scene):

    def construct(self):
        ofcourse = TexText("Of course $\\cdots$")
        self.play(
            Write(ofcourse)
        )
        self.wait(4)
        title = VGroup(
            TexText("Take these home!").set_color(YELLOW),
            Underline(TexText("Take these home!")).set_color(YELLOW)
        )
        title.to_edge(UP)

        self.play(
            ReplacementTransform(ofcourse, title)
        )
        self.wait(2)

        pt1 = VGroup(
            Dot(),
            TexText("Models approximate reality")
        ).arrange()

        pt2 = VGroup(
            Dot(),
            TexText("We construct models from informed guesses")
        ).arrange()

        pt3 = VGroup(
            Dot(),
            TexText("Model must agree with experiments/observations")
        ).arrange()

        # pt4 = VGroup(
        #     Dot(),
        #     TexText("Foxes doesn't have any other food source")
        # ).arrange()

        points = VGroup(pt1, pt2, pt3).arrange_in_grid(3, 1, aligned_edge=LEFT, buff=1.0)
        points.to_edge(LEFT)
        points.fade(0.9)
        pt1.fade(0.0)
        self.play(
            FadeIn(points)
        )

        self.wait(3)

        self.play(
            pt1.animate.fade(0.9),
            pt2.animate.fade(0.0)
        )

        self.wait(3)

        self.play(
            pt2.animate.fade(0.9),
            pt3.animate.fade(0.0)
        )

        self.wait(3)

        # self.play(
        #     pt3.animate.fade(0.9),
        #     pt4.animate.fade(0.0)
        # )

        self.wait(3)

class Thanks(Scene):

    def construct(self):
        vidicon = ImageMobject("video_icon.jpg").scale(0.7)
        vidicon.shift(LEFT * 3)
        sciencesort = Text("science.sort()")
        sciencesort.next_to(vidicon)
        Thanks = TexText("Thanks for tuning in!!").scale(2)
        self.play(
            FadeInFromPoint(Thanks, Thanks.get_bottom() + DOWN)
        )
        # self.wait(2)
        
        formore = Group(
            Text("**Check out"),
            vidicon,
            sciencesort,
            Text("  for more!!")
        ).arrange().scale(0.3).to_corner(DR)
        self.add(formore)
        self.wait()