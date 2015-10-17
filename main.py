# OpenNero will execute ModMain when this mod is loaded
import Maze.main
from Maze.module import getMod, MazeEnvironment, GranularMazeEnvironment
import Hw3.agent

def start_my_tabular_agent():
    """ start my tabular RL agent """
    getMod().start_agent("data/shapes/character/MyTabularRLRobot.xml", MazeEnvironment)

def start_my_tabular_agent_granular():
    """ start my tabular RL agent in the fine-grained maze """
    getMod().start_agent("data/shapes/character/MyTabularRLRobot.xml", GranularMazeEnvironment)

def start_my_tiling_agent_granular():
    """ start my tiling RL agent in the fine-grained maze """
    getMod().start_agent("data/shapes/character/MyTilingRLRobot.xml", GranularMazeEnvironment)

def start_my_nearest_neighbors_agent_granular():
    """ start my nearest neighbors RL agent in the fine-grained maze """
    getMod().start_agent("data/shapes/character/MyNearestNeighborsRLRobot.xml", GranularMazeEnvironment)

def ModMain():
    agent_descs = [
            ('My Tabular RL Agent (Coarse)', start_my_tabular_agent, True, ""),
            ('My Tabular RL Agent (Fine)', start_my_tabular_agent_granular, True, ""),
            ('My Tiling RL Agent (Fine)', start_my_tiling_agent_granular, True, ""),
            ('My Nearest Neighbors RL Agent (Fine)', start_my_nearest_neighbors_agent_granular, True, ""),
    ]
    getMod().AGENTS.extend(agent_descs)
    Maze.main.ModMain()
