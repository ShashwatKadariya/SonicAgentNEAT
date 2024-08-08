import retro
import neat
import cv2
import numpy as np
import pickle
import os

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        observation = env.reset()


        x, y, z = env.observation_space.shape

        v = lambda x: int(x / 8)
        x, y = v(x), v(y)

        net = neat.nn.FeedForwardNetwork.create(genome=genome, config=config)

        xpos, xposMax, fitness, counter, fitnessMax= 0, 0, 0, 0, 0

        done = False
        renderGame = True
        cv2.namedWindow("sonic-EYE", cv2.WINDOW_NORMAL)
        while not done:
            imgArray = []
            if renderGame:
                env.render()

            scaledIMG = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            scaledIMG = cv2.resize(scaledIMG, (x, y))

            observation = cv2.resize(observation, (x, y))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) 
            observation = np.reshape(observation, (x, y))

            cv2.imshow('sonic-EYE', scaledIMG)
            cv2.waitKey(1)

            imgArray = np.ndarray.flatten(observation)

            action = net.activate(imgArray)
            observation, reward, done, info = env.step(action)



            xpos = info['x']

            if xpos > xposMax:
                xposMax = xpos
                fitness += reward
                counter = 0
            else:
                counter += 1

            if counter > 250:
                done = True

            genome.fitness = fitness

            if (xpos >= info['screen_x_end']) and (info['x'] > 400):
                done = True
                fitness += 100000

            print(fitness)


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(15))

    winner = p.run(eval_genomes)

    return winner

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-network')

    env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1")

    winner = run(config_file= config_path)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
    output.close()