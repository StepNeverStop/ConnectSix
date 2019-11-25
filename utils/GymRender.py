from gym.envs.classic_control import rendering


class GymRender(object):

    def __init__(self):
        self.viewer = None

    def render(self, game, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        gap = 30
        screen_width = (game.dim + 1) * gap
        screen_height = (game.dim + 1) * gap

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        self.viewer.geoms.clear()

        for i in range(1, game.dim + 1):
            self.track = rendering.Line((gap, i * gap), (screen_width - gap, i * gap))
            self.track.set_color(0.7, 0.7, 0.7)
            self.viewer.add_geom(self.track)

        for i in range(1, game.dim + 1):
            self.track = rendering.Line((i * gap, gap), (i * gap, screen_height - gap))
            self.track.set_color(0.7, 0.7, 0.7)
            self.viewer.add_geom(self.track)

        for x in range(game.dim):
            for y in range(game.dim):
                stone = game.board[x][y]
                _y = game.dim - x - 1
                _x = y
                if stone == 0:
                    self.axle = rendering.make_circle(gap / 3)
                    self.axle.add_attr(rendering.Transform(translation=((_x + 1) * gap, (_y + 1) * gap)))
                    self.axle.set_color(0, 0, 0)
                    self.viewer.add_geom(self.axle)
                elif stone == 1:
                    self.axle = rendering.make_circle(gap / 3)
                    self.axle.add_attr(rendering.Transform(translation=((_x + 1) * gap, (_y + 1) * gap)))
                    self.axle.set_color(1, 1, 1)
                    self.viewer.add_geom(self.axle)
                    self.axle = rendering.make_circle(gap / 3, filled=False)
                    self.axle.add_attr(rendering.Transform(translation=((_x + 1) * gap, (_y + 1) * gap)))
                    self.axle.set_color(0, 0, 0)
                    self.viewer.add_geom(self.axle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
