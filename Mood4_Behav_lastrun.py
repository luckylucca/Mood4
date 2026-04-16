#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Thu Apr 16 17:01:29 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from initialize_variables
import numpy as np # Keep in mind that by default, numpy version is 1.26
import time
import sys

# Set happiness trials
# numBlocks = 4
# happyTrialIdx = np.ceil(np.linspace(2, 36*numBlocks, 12*numBlocks))
# print(sys.version)

start_money = 5 # Start amount
option_size = 0.2
width = 0.335
height = 0.2
time_first_delay = 2
time_gamble_show = 1 # in seconds
time_gamble_delay = 0.5
time_chosen_gamble = 1
time_gamble_result = 2 # time after which they go to the next trial
time_iti = 1

numBlocks = 4
happyTrialIdx = np.hstack((3*np.ones(10*numBlocks), 2*np.ones(10*numBlocks)))
happyTrialIdx = happyTrialIdx[np.random.permutation(len(happyTrialIdx))]
happyTrialIdx = np.cumsum(happyTrialIdx)-1

happyTrial = np.zeros(50*numBlocks)
for i in happyTrialIdx:
    happyTrial[int(i)] = 1

happyTrial = (happyTrial != 0)


#happyTrialIdx_prac = np.ceil(np.linspace(2, 9*10, 3*10))
happyTrialIdx_prac = np.hstack((3*np.ones(2*10), 2*np.ones(2*10)))
happyTrialIdx_prac = happyTrialIdx_prac[np.random.permutation(len(happyTrialIdx_prac))]
happyTrialIdx_prac = np.cumsum(happyTrialIdx_prac)-1

happyTrial_prac = np.zeros(100)
for i in happyTrialIdx_prac:
    happyTrial_prac[int(i)] = 1

happyTrial_prac = (happyTrial_prac != 0)


npractice_trials = 10   # Number of practice trails
ntrials        = 192    # Number of trails
#ntotal_trials  = ntrials + (npractice_trials*3) # Allow max 3 reapeats of the practice trials
#reward         = np.linspace(0, 0.5, 11) # All possible reward amounts
#proba_vec      = np.random.randint(51, size=(ntotal_trials,1))*2 # Generate the probabilities for each ntrials
#divider        = np.ceil(ntotal_trials*2/len(reward))
#reward_vec     = np.repeat(reward, divider).reshape(-1,1)
#reward_vec     = np.random.permutation(reward_vec)
#sure_option    = reward_vec[0:ntotal_trials]
#gambles        = np.concatenate((reward_vec[ntotal_trials:ntotal_trials*2], proba_vec), axis=1)
#gamble_result  = np.random.randint(101, size=(ntotal_trials,1))
#sure_side      = np.random.randint(2, size=(ntotal_trials,1))

reward_vec = np.array([[0.15, 1.00, 0.00, 0.50, 0.25, 0.50],
 [0.15, 1.00, 0.00, 0.70, 0.25, 0.30],
 [0.15, 1.00, 0.00, 0.80, 0.25, 0.20],
 [0.15, 1.00, 0.00, 0.40, 0.30, 0.60],
 [0.15, 1.00, 0.00, 0.60, 0.30, 0.40],
 [0.15, 1.00, 0.00, 0.80, 0.30, 0.20],
 [0.25, 1.00, 0.00, 0.30, 0.35, 0.70],
 [0.25, 1.00, 0.00, 0.50, 0.35, 0.50],
 [0.25, 1.00, 0.00, 0.70, 0.35, 0.30],
 [0.35, 1.00, 0.00, 0.20, 0.40, 0.80],
 [0.35, 1.00, 0.00, 0.30, 0.40, 0.70],
 [0.35, 1.00, 0.00, 0.50, 0.40, 0.50],
 [0.15, 1.00, 0.00, 0.30, 0.50, 0.70],
 [0.15, 1.00, 0.00, 0.60, 0.50, 0.40],
 [0.15, 1.00, 0.00, 0.70, 0.50, 0.30],
 [0.15, 1.00, 0.00, 0.80, 0.50, 0.20],
 [0.15, 1.00, 0.00, 0.90, 0.50, 0.10],
 [0.25, 1.00, 0.00, 0.10, 0.50, 0.90],
 [0.25, 1.00, 0.00, 0.30, 0.50, 0.70],
 [0.25, 1.00, 0.00, 0.50, 0.50, 0.50],
 [0.25, 1.00, 0.00, 0.60, 0.50, 0.40],
 [0.25, 1.00, 0.00, 0.70, 0.50, 0.30],
 [0.25, 1.00, 0.00, 0.90, 0.50, 0.10],
 [0.35, 1.00, 0.00, 0.10, 0.50, 0.90],
 [0.35, 1.00, 0.00, 0.30, 0.50, 0.70],
 [0.35, 1.00, 0.00, 0.50, 0.50, 0.50],
 [0.35, 1.00, 0.00, 0.70, 0.50, 0.30],
 [0.25, 1.00, 0.00, 0.00, 0.00, 0.00],
 [0.00, 1.00, 0.00, 0.70, 0.25, 0.30],
 [0.00, 1.00, 0.00, 0.10, 0.30, 0.90],
 [0.00, 1.00, 0.00, 0.70, 0.40, 0.30],
 [0.00, 1.00, 0.00, 0.50, 0.35, 0.50],
 [0.00, 1.00, 0.00, 0.50, 0.50, 0.50]])

options_vec_reversed = reward_vec[:,[2, 3, 4, 5, 0, 1]]
options1 = np.random.permutation(np.vstack((reward_vec,options_vec_reversed)))
options2 = np.random.permutation(np.vstack((reward_vec,options_vec_reversed)))
options3 = np.random.permutation(np.vstack((reward_vec,options_vec_reversed)))
options_prac = np.random.permutation(np.vstack((reward_vec,options_vec_reversed)))[1:30]
options = np.vstack((options1, options2, options3)) # All this is done to respect the randomization of the task

#gambles_txt = []
#for n in range(ntotal_trials):
#    gamble1 = f"{gambles[n, 0]:.0f}%"  # Gamble1
#    if gambles[n, 1] == reward[2]:
#        gamble2 = f"BIG"
##        gamble2 = f"BIG (${gambles[n, 1]:.2f})"
#    elif gambles[n, 1] == reward[1]:
#        gamble2 = f"MEDIUM"
##        gamble2 = f"MEDIUM (${gambles[n, 1]:.2f})"
#    else:
#        gamble2 = f"SMALL"
##        gamble2 = f"SMALL (${gambles[n, 1]:.2f})"
#        
#    gamble3 = f"{gambles[n, 2]:.0f}%"  # Gamble2
#    if gambles[n, 3] == reward[2]:
#        gamble4 = f"BIG"
##        gamble4 = f"BIG (${gambles[n, 3]:.2f})"
#    elif gambles[n, 3] == reward[1]:
#        gamble4 = f"MEDIUM"
#        gamble4 = f"MEDIUM (${gambles[n, 3]:.2f})"
#    else:
#        gamble4 = f"SMALL"
##        gamble4 = f"SMALL (${gambles[n, 3]:.2f})"
#    
#    gambles_txt.append([gamble1, gamble2, gamble3, gamble4])
    
current_money = start_money

#thisExp.addData("Gambles", gambles[3*npractice_trials:][:]) No need: each gamble is saved to the exp on each trial loop
#thisExp.addData("Practice gamble", gambles[0:3*npractice_trials][:])
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'Mood4'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'PROLIFIC_PID': '',
    'STUDY_ID': '',
    'SESSION_ID': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2560, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['PROLIFIC_PID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/lucas/Documents/Experiments/Mood4/github_Mood4/Mood4_Behav_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0.3,0.3,0.3], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0.3,0.3,0.3]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('choice_inst_5a') is None:
        # initialise choice_inst_5a
        choice_inst_5a = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='choice_inst_5a',
        )
    if deviceManager.getDevice('intro_next_trial_input_2') is None:
        # initialise intro_next_trial_input_2
        intro_next_trial_input_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_next_trial_input_2',
        )
    if deviceManager.getDevice('choice_inst_5b') is None:
        # initialise choice_inst_5b
        choice_inst_5b = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='choice_inst_5b',
        )
    if deviceManager.getDevice('intro_next_trial_input') is None:
        # initialise intro_next_trial_input
        intro_next_trial_input = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_next_trial_input',
        )
    if deviceManager.getDevice('key_resp_inst_5') is None:
        # initialise key_resp_inst_5
        key_resp_inst_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_inst_5',
        )
    if deviceManager.getDevice('key_resp_inst_6') is None:
        # initialise key_resp_inst_6
        key_resp_inst_6 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_inst_6',
        )
    if deviceManager.getDevice('key_resp_inst_7') is None:
        # initialise key_resp_inst_7
        key_resp_inst_7 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_inst_7',
        )
    if deviceManager.getDevice('key_resp_7') is None:
        # initialise key_resp_7
        key_resp_7 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_7',
        )
    if deviceManager.getDevice('key_resp_inst_8') is None:
        # initialise key_resp_inst_8
        key_resp_inst_8 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_inst_8',
        )
    if deviceManager.getDevice('key_resp_8') is None:
        # initialise key_resp_8
        key_resp_8 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_8',
        )
    if deviceManager.getDevice('key_resp_inst_11') is None:
        # initialise key_resp_inst_11
        key_resp_inst_11 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_inst_11',
        )
    if deviceManager.getDevice('key_resp_inst_9') is None:
        # initialise key_resp_inst_9
        key_resp_inst_9 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_inst_9',
        )
    if deviceManager.getDevice('p_choice') is None:
        # initialise p_choice
        p_choice = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='p_choice',
        )
    if deviceManager.getDevice('p_next_trial_input') is None:
        # initialise p_next_trial_input
        p_next_trial_input = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='p_next_trial_input',
        )
    if deviceManager.getDevice('key_resp_5') is None:
        # initialise key_resp_5
        key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_5',
        )
    if deviceManager.getDevice('key_resp_9') is None:
        # initialise key_resp_9
        key_resp_9 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_9',
        )
    if deviceManager.getDevice('key_resp_inst_10') is None:
        # initialise key_resp_inst_10
        key_resp_inst_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_inst_10',
        )
    if deviceManager.getDevice('gamblechoice') is None:
        # initialise gamblechoice
        gamblechoice = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='gamblechoice',
        )
    if deviceManager.getDevice('next_trial_input') is None:
        # initialise next_trial_input
        next_trial_input = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='next_trial_input',
        )
    if deviceManager.getDevice('key_resp_6') is None:
        # initialise key_resp_6
        key_resp_6 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_6',
        )
    if deviceManager.getDevice('key_resp_10') is None:
        # initialise key_resp_10
        key_resp_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_10',
        )
    if deviceManager.getDevice('key_resp_11') is None:
        # initialise key_resp_11
        key_resp_11 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_11',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "consent" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text='This is an academic research project conducted through the University of Pennsylvania. In this Happiness Task, you will play a decision making game. In this game, you will make decisions about two options presented before you. The task takes ~35 minutes. You must be at least 18 years old to participate. Your participation in this research is voluntary, which means you can choose whether or not to participate without adverse consequences. Your anonymity is assured: the researchers who have requested your participation will not receive any personal information about you except your worker ID, gender, and age. The worker IDs will not be shared with anyone outside the research team nor associated with the collected data. The de-identified data may be stored and distributed for future research studies without additional informed consent. If you have questions about this research, please contact Nicole Rust by emailing nrust@psych.upenn.edu. If you have questions, concerns, or complaints regarding your participation in this research study, or if you have any questions about your rights as a research subject and you cannot reach a member of the study team, you may contact the Office of Regulatory Affairs at the University of Pennsylvania by calling (215) 898-2614 or emailing irb@pobox.upenn.edu.\n\nBy clicking this box below you acknowledge that you have read the above details and consent to participate in this study.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button = visual.ButtonStim(win, 
        text='I agree', font='Arvo',
        pos=(0, -0.45),
        letterHeight=0.025,
        size=(0.4, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='green', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-1
    )
    button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "instructions1" ---
    Intro_text_show = visual.TextStim(win=win, name='Intro_text_show',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Continue_txt = visual.TextStim(win=win, name='Continue_txt',
        text='Press <enter> to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from initialize_variables
    startTime = time.strftime("%Y-%m-%d_%H.%M.%S")
    thisExp.addData("dateTime", startTime) 
    
    # --- Initialize components for Routine "instructions_t1_target_on_choice" ---
    fixation_inst_5a = visual.ShapeStim(
        win=win, name='fixation_inst_5a', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    intro_sure = visual.Rect(
        win=win, name='intro_sure',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(-width, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    intro_sure_txt = visual.TextStim(win=win, name='intro_sure_txt',
        text='',
        font='Arial',
        pos=(-width, 0), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_gambletop = visual.Rect(
        win=win, name='intro_gambletop',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(width, height), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    intro_gambletop_txt = visual.TextStim(win=win, name='intro_gambletop_txt',
        text='',
        font='Arial',
        pos=(width, height + 0.03), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    intro_gambletop_p = visual.TextStim(win=win, name='intro_gambletop_p',
        text='',
        font='Arial',
        pos=(width, height-0.04), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    intro_gamblelow = visual.Rect(
        win=win, name='intro_gamblelow',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(width, -height), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    intro_gamblelow_txt = visual.TextStim(win=win, name='intro_gamblelow_txt',
        text='0',
        font='Arial',
        pos=(width, -height+0.03), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    intro_gamblelow_p = visual.TextStim(win=win, name='intro_gamblelow_p',
        text='20%',
        font='Arial',
        pos=(width, -height-0.04), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    text_inst_5a = visual.TextStim(win=win, name='text_inst_5a',
        text='On this trial you can either choose a certain reward of 0.5$ or a gamble with a 80% chance of getting a 0.25$ reward.\nPress the <left arrow key> to select the certain reward of 0.5$.',
        font='Arial',
        pos=(0, -0.43), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    choice_inst_5a = keyboard.Keyboard(deviceName='choice_inst_5a')
    
    # --- Initialize components for Routine "instructions_t1_chosen_gamble" ---
    fixation_inst_5a_2 = visual.ShapeStim(
        win=win, name='fixation_inst_5a_2', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    intro_sure_2 = visual.Rect(
        win=win, name='intro_sure_2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(-width, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    intro_sure_txt_2 = visual.TextStim(win=win, name='intro_sure_txt_2',
        text='',
        font='Arial',
        pos=(-width, 0), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "instructions_t1_outcome" ---
    total_prompt_inst_5a = visual.TextStim(win=win, name='total_prompt_inst_5a',
        text='Current total: ',
        font='Arial',
        pos=(-0.55, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    curr_total_inst_5a = visual.TextStim(win=win, name='curr_total_inst_5a',
        text='',
        font='Arial',
        pos=(-0.3, 0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    p_outcome_square_2 = visual.Rect(
        win=win, name='p_outcome_square_2',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    intro_outcome_text = visual.TextStim(win=win, name='intro_outcome_text',
        text='+0.50$',
        font='Arial',
        pos=(-width, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_next_trial_txt_2 = visual.TextStim(win=win, name='intro_next_trial_txt_2',
        text='Press the <space bar> to do another introductory trial',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    intro_next_trial_input_2 = keyboard.Keyboard(deviceName='intro_next_trial_input_2')
    
    # --- Initialize components for Routine "intro_iti" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "instructions_t2_target_on_choice" ---
    fixation_inst_5b = visual.ShapeStim(
        win=win, name='fixation_inst_5b', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    intro_sure_3 = visual.Rect(
        win=win, name='intro_sure_3',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(-width, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    intro_gambletop_2 = visual.Rect(
        win=win, name='intro_gambletop_2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(width, height), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    intro_gamblelow_2 = visual.Rect(
        win=win, name='intro_gamblelow_2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(width, -height), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    intro_sure_txt_3 = visual.TextStim(win=win, name='intro_sure_txt_3',
        text='',
        font='Arial',
        pos=(-width, 0), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    intro_gambletop_txt_2 = visual.TextStim(win=win, name='intro_gambletop_txt_2',
        text='',
        font='Arial',
        pos=(width, height + 0.03), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    intro_gambletop_p_2 = visual.TextStim(win=win, name='intro_gambletop_p_2',
        text='',
        font='Arial',
        pos=(width, height-0.04), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    intro_gamblelow_txt_2 = visual.TextStim(win=win, name='intro_gamblelow_txt_2',
        text='0',
        font='Arial',
        pos=(width, -height+0.03), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    intro_gamblelow_p_2 = visual.TextStim(win=win, name='intro_gamblelow_p_2',
        text='20%',
        font='Arial',
        pos=(width, -height-0.04), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    text_inst_5b = visual.TextStim(win=win, name='text_inst_5b',
        text='On this trial you can either choose to win a $0.15 reward for certain, or choose a 70% chance of getting a $0.50 reward.\n\nPress the <right arrow key> to select the gamble.',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    choice_inst_5b = keyboard.Keyboard(deviceName='choice_inst_5b')
    
    # --- Initialize components for Routine "instructions_t2_chosen_gamble" ---
    fixation_inst_5a_3 = visual.ShapeStim(
        win=win, name='fixation_inst_5a_3', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    intro_gambletop_3 = visual.Rect(
        win=win, name='intro_gambletop_3',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(width, height), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    intro_gamblelow_3 = visual.Rect(
        win=win, name='intro_gamblelow_3',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=(width, -height), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    intro_gambletop_txt_3 = visual.TextStim(win=win, name='intro_gambletop_txt_3',
        text='',
        font='Arial',
        pos=(width, height + 0.03), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_gambletop_p_3 = visual.TextStim(win=win, name='intro_gambletop_p_3',
        text='',
        font='Arial',
        pos=(width, height-0.04), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    intro_gamblelow_txt_3 = visual.TextStim(win=win, name='intro_gamblelow_txt_3',
        text='0',
        font='Arial',
        pos=(width, -height+0.03), draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    intro_gamblelow_p_3 = visual.TextStim(win=win, name='intro_gamblelow_p_3',
        text='20%',
        font='Arial',
        pos=(width, -height-0.04), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "instructions_t2_outcome" ---
    total_prompt_inst_5a_2 = visual.TextStim(win=win, name='total_prompt_inst_5a_2',
        text='Current total: ',
        font='Arial',
        pos=(-0.55, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    curr_total_inst_5a_2 = visual.TextStim(win=win, name='curr_total_inst_5a_2',
        text='',
        font='Arial',
        pos=(-0.3, 0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 0.6863, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    p_outcome_square_3 = visual.Rect(
        win=win, name='p_outcome_square_3',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    intro_outcome_text_3 = visual.TextStim(win=win, name='intro_outcome_text_3',
        text='Loss!',
        font='Arial',
        pos=(width, 0.03), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_next_trial_txt = visual.TextStim(win=win, name='intro_next_trial_txt',
        text='Press the <space bar> to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    intro_next_trial_input = keyboard.Keyboard(deviceName='intro_next_trial_input')
    
    # --- Initialize components for Routine "instructions6" ---
    instructions_text = visual.TextStim(win=win, name='instructions_text',
        text='On the gamble outcome screen, the top of the screen will display the total amount of money  you earned, and your progress throughout the task will be displayed on a progress bar at the  bottom. The next screen will show you an example.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Continue_txt_2 = visual.TextStim(win=win, name='Continue_txt_2',
        text='Press <enter> to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_inst_5 = keyboard.Keyboard(deviceName='key_resp_inst_5')
    
    # --- Initialize components for Routine "instructions7" ---
    fixation_inst_5a_6 = visual.ShapeStim(
        win=win, name='fixation_inst_5a_6', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    total_prompt_inst = visual.TextStim(win=win, name='total_prompt_inst',
        text='Current total: ',
        font='Arial',
        pos=(-0.55, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    curr_total_inst = visual.TextStim(win=win, name='curr_total_inst',
        text='$5',
        font='Arial',
        pos=(-0.3, 0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 0.6863, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    p_outcome_square_4 = visual.Rect(
        win=win, name='p_outcome_square_4',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    intro_outcome_text_5 = visual.TextStim(win=win, name='intro_outcome_text_5',
        text='Loss!',
        font='Arial',
        pos=(-width, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_inst_6 = keyboard.Keyboard(deviceName='key_resp_inst_6')
    instrutions_text = visual.TextStim(win=win, name='instrutions_text',
        text='This is an example outcome screen for a\nloss, also showing your current total\nand the progress bar (about half way \nthrough the session).\n\nPress <enter> to continue',
        font='Arial',
        pos=(0.45, 0.1), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='blue', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    prog_5 = visual.Progress(
        win, name='prog_5',
        progress=0.55,
        pos=(-0.5, -0.45), size=(1, 0.03), anchor='center-left', units='height',
        barColor='white', backColor=None, borderColor='white', colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-7
    )
    arrow1 = visual.ShapeStim(
        win=win, name='arrow1', vertices='arrow',
        size=(0.1, 0.2),
        ori=-60.0, pos=(-0.2, 0.3), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='blue', fillColor='blue',
        opacity=None, depth=-8.0, interpolate=True)
    arrow2 = visual.ShapeStim(
        win=win, name='arrow2', vertices='arrow',
        size=(0.1, 0.2),
        ori=-160.0, pos=(0.05, -0.3), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='blue', fillColor='blue',
        opacity=None, depth=-9.0, interpolate=True)
    
    # --- Initialize components for Routine "instructions8" ---
    instructions_text_3 = visual.TextStim(win=win, name='instructions_text_3',
        text='At various points throughout the task you will be asked to rate your happiness. When you are asked to make a happiness rating, click a number 1-20 to reflect your rating from 1 (very unhappy) to 20 (very happy). ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Continue_txt_3 = visual.TextStim(win=win, name='Continue_txt_3',
        text='Press <enter> to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_inst_7 = keyboard.Keyboard(deviceName='key_resp_inst_7')
    
    # --- Initialize components for Routine "happy_overall" ---
    question_3 = visual.TextStim(win=win, name='question_3',
        text='Taken together, how happy are you right now overall? ',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    happiness_rating_overall = visual.Slider(win=win, name='happiness_rating_overall',
        startValue=None, size=(1.3, 0.05), pos=(0, 0), units=win.units,
        labels=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'), ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    low_end_text_3 = visual.TextStim(win=win, name='low_end_text_3',
        text='very unhappy',
        font='Arial',
        pos=(-0.5, -0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    high_end_text_3 = visual.TextStim(win=win, name='high_end_text_3',
        text='very happy',
        font='Arial',
        pos=(0.5, -0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_7 = keyboard.Keyboard(deviceName='key_resp_7')
    exit_text_4 = visual.TextStim(win=win, name='exit_text_4',
        text='Click to enter your answer. Press <enter> to move to the next screen.',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "instructions10" ---
    instructions_text_6 = visual.TextStim(win=win, name='instructions_text_6',
        text='Just now, you were asked to think about your life overall. Now, think about just right now. How happy are you at this moment?\n\nDuring the task you will be asked this question many times. It is VERY important that you use as much of the rating scale as you can. \n\nIn a moment you will complete some practice trials. The least happy you remember being during the practice session should correspond to somewhere in the lower half of the rating scale. The most happy you remember being should correspond to somewhere in the upper half.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Continue_txt_4 = visual.TextStim(win=win, name='Continue_txt_4',
        text='Press <enter> to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_inst_8 = keyboard.Keyboard(deviceName='key_resp_inst_8')
    
    # --- Initialize components for Routine "happy_base" ---
    happiness_rating_baseline = visual.Slider(win=win, name='happiness_rating_baseline',
        startValue=None, size=(1.3, 0.05), pos=(0, 0), units=win.units,
        labels=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'), ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=0, readOnly=False)
    low_end_text_5 = visual.TextStim(win=win, name='low_end_text_5',
        text='very unhappy',
        font='Arial',
        pos=(-0.5, -0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    high_end_text_5 = visual.TextStim(win=win, name='high_end_text_5',
        text='very happy',
        font='Arial',
        pos=(0.5, -0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    question_5 = visual.TextStim(win=win, name='question_5',
        text='How happy are you at this moment?',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    exit_text_3 = visual.TextStim(win=win, name='exit_text_3',
        text='Click to select your answer. Press <enter> to continue to the next screen.',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_8 = keyboard.Keyboard(deviceName='key_resp_8')
    
    # --- Initialize components for Routine "instructions10_2" ---
    instructions_text_2 = visual.TextStim(win=win, name='instructions_text_2',
        text='It would be best to complete the entire task with minimal breaks. If you need to take a break, please do so on the break screen.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Continue_txt_6 = visual.TextStim(win=win, name='Continue_txt_6',
        text='Press <enter> to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_inst_11 = keyboard.Keyboard(deviceName='key_resp_inst_11')
    
    # --- Initialize components for Routine "instructions11" ---
    instructions_text_7 = visual.TextStim(win=win, name='instructions_text_7',
        text='You will now complete 10 practice trials. Try to remember the least and most happy you feel during the practice trials to guide your happiness ratings throughout the task.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Continue_txt_5 = visual.TextStim(win=win, name='Continue_txt_5',
        text='Press <enter> to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_inst_9 = keyboard.Keyboard(deviceName='key_resp_inst_9')
    
    # --- Initialize components for Routine "first_delay" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "p_options_show" ---
    p_fixation_cross_4 = visual.ShapeStim(
        win=win, name='p_fixation_cross_4', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    p_box1 = visual.Rect(
        win=win, name='p_box1',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    p_box2 = visual.Rect(
        win=win, name='p_box2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    p_box3 = visual.Rect(
        win=win, name='p_box3',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    p_box4 = visual.Rect(
        win=win, name='p_box4',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    p_box1_mag = visual.TextStim(win=win, name='p_box1_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    p_box1_P = visual.TextStim(win=win, name='p_box1_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    p_box2_mag = visual.TextStim(win=win, name='p_box2_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    p_box2_P = visual.TextStim(win=win, name='p_box2_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    p_box3_mag = visual.TextStim(win=win, name='p_box3_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    p_box3_P = visual.TextStim(win=win, name='p_box3_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    p_box4_mag = visual.TextStim(win=win, name='p_box4_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    p_box4_P = visual.TextStim(win=win, name='p_box4_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    p_choice = keyboard.Keyboard(deviceName='p_choice')
    
    # --- Initialize components for Routine "p_chosen_option" ---
    fixation_5 = visual.ShapeStim(
        win=win, name='fixation_5', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, 1.0000, 1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    p_box1_2 = visual.Rect(
        win=win, name='p_box1_2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    p_box2_2 = visual.Rect(
        win=win, name='p_box2_2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    p_box3_2 = visual.Rect(
        win=win, name='p_box3_2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    p_box4_2 = visual.Rect(
        win=win, name='p_box4_2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    p_box1_mag_2 = visual.TextStim(win=win, name='p_box1_mag_2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    p_box1_P_2 = visual.TextStim(win=win, name='p_box1_P_2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    p_box2_mag_2 = visual.TextStim(win=win, name='p_box2_mag_2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    p_box2_P_2 = visual.TextStim(win=win, name='p_box2_P_2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "p_outcome" ---
    # Run 'Begin Experiment' code from p_gamble_result_code
    p_progVal = 1
    fixation_6 = visual.ShapeStim(
        win=win, name='fixation_6', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, 1.0000, 1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    p_money_prompt = visual.TextStim(win=win, name='p_money_prompt',
        text='Current total: ',
        font='Arial',
        pos=(-0.55, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    p_current_money_txt = visual.TextStim(win=win, name='p_current_money_txt',
        text='',
        font='Arial',
        pos=(-0.3, 0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 0.6863, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    practice_txt = visual.TextStim(win=win, name='practice_txt',
        text='Practice',
        font='Arial',
        pos=(-0.55, 0.33), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    p_outcome_square = visual.Rect(
        win=win, name='p_outcome_square',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    p_outcome_text = visual.TextStim(win=win, name='p_outcome_text',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    p_outcome_rewa_text = visual.TextStim(win=win, name='p_outcome_rewa_text',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    p_prog_bar = visual.Progress(
        win, name='p_prog_bar',
        progress=0.0,
        pos=(-0.5, -0.45), size=(1, 0.03), anchor='center-left', units='height',
        barColor='white', backColor=None, borderColor=None, colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-8
    )
    p_next_trial_txt = visual.TextStim(win=win, name='p_next_trial_txt',
        text='Press the <space bar> to initiate the next trial. ',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    p_next_trial_input = keyboard.Keyboard(deviceName='p_next_trial_input')
    
    # --- Initialize components for Routine "iti" ---
    text = visual.TextStim(win=win, name='text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "p_happiness_trial" ---
    happiness_rating_prac = visual.Slider(win=win, name='happiness_rating_prac',
        startValue=None, size=(1.3, 0.05), pos=(0, 0), units=win.units,
        labels=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'), ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=0, readOnly=False)
    low_end_text_4 = visual.TextStim(win=win, name='low_end_text_4',
        text='very unhappy',
        font='Arial',
        pos=(-0.5, -0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    high_end_text_4 = visual.TextStim(win=win, name='high_end_text_4',
        text='very happy',
        font='Arial',
        pos=(0.5, -0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    question_4 = visual.TextStim(win=win, name='question_4',
        text='How happy are you at this moment?',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    exit_text_2 = visual.TextStim(win=win, name='exit_text_2',
        text='Press <enter> to move to the next trial',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_5 = keyboard.Keyboard(deviceName='key_resp_5')
    
    # --- Initialize components for Routine "p_h_iti" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "repeat_practice" ---
    repeat_practice_txt = visual.TextStim(win=win, name='repeat_practice_txt',
        text="You have now completed the practice trials. \n\nPress 'c' to continue to the main trials.\n\nPress 'r' to repeat the practice trials.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_9 = keyboard.Keyboard(deviceName='key_resp_9')
    
    # --- Initialize components for Routine "repeat_iti" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "instructions12" ---
    instructions_text_8 = visual.TextStim(win=win, name='instructions_text_8',
        text='You have now completed the practice trials and will move on to the full task. \n\nPress <enter> to start',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_inst_10 = keyboard.Keyboard(deviceName='key_resp_inst_10')
    
    # --- Initialize components for Routine "first_delay" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "gamble1_show" ---
    fixation_cross = visual.ShapeStim(
        win=win, name='fixation_cross', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    gamble1_rew_text = visual.TextStim(win=win, name='gamble1_rew_text',
        text='',
        font='Arial',
        pos=(-width, reward[2]+rew_txt_margin+0.03), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    gamble1_rew_text_low = visual.TextStim(win=win, name='gamble1_rew_text_low',
        text='',
        font='Arial',
        pos=(-width, reward[2]+rew_txt_margin), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    gamble1_prob_text = visual.TextStim(win=win, name='gamble1_prob_text',
        text='',
        font='Arial',
        pos=(-width, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "p_options_show" ---
    p_fixation_cross_4 = visual.ShapeStim(
        win=win, name='p_fixation_cross_4', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    p_box1 = visual.Rect(
        win=win, name='p_box1',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    p_box2 = visual.Rect(
        win=win, name='p_box2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    p_box3 = visual.Rect(
        win=win, name='p_box3',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    p_box4 = visual.Rect(
        win=win, name='p_box4',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    p_box1_mag = visual.TextStim(win=win, name='p_box1_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    p_box1_P = visual.TextStim(win=win, name='p_box1_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    p_box2_mag = visual.TextStim(win=win, name='p_box2_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    p_box2_P = visual.TextStim(win=win, name='p_box2_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    p_box3_mag = visual.TextStim(win=win, name='p_box3_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    p_box3_P = visual.TextStim(win=win, name='p_box3_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    p_box4_mag = visual.TextStim(win=win, name='p_box4_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    p_box4_P = visual.TextStim(win=win, name='p_box4_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    p_choice = keyboard.Keyboard(deviceName='p_choice')
    
    # --- Initialize components for Routine "gamble2_show" ---
    fixation_cross_2 = visual.ShapeStim(
        win=win, name='fixation_cross_2', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    gamble2_rew_text = visual.TextStim(win=win, name='gamble2_rew_text',
        text='',
        font='Arial',
        pos=(width, reward[2]+rew_txt_margin+0.03), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    gamble2_rew_text_low = visual.TextStim(win=win, name='gamble2_rew_text_low',
        text='',
        font='Arial',
        pos=(width, reward[2]+rew_txt_margin), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    gamble2_prob_text = visual.TextStim(win=win, name='gamble2_prob_text',
        text='',
        font='Arial',
        pos=(width, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "p_options_show" ---
    p_fixation_cross_4 = visual.ShapeStim(
        win=win, name='p_fixation_cross_4', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    p_box1 = visual.Rect(
        win=win, name='p_box1',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    p_box2 = visual.Rect(
        win=win, name='p_box2',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    p_box3 = visual.Rect(
        win=win, name='p_box3',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    p_box4 = visual.Rect(
        win=win, name='p_box4',
        width=(option_size, option_size)[0], height=(option_size, option_size)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    p_box1_mag = visual.TextStim(win=win, name='p_box1_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    p_box1_P = visual.TextStim(win=win, name='p_box1_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    p_box2_mag = visual.TextStim(win=win, name='p_box2_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    p_box2_P = visual.TextStim(win=win, name='p_box2_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    p_box3_mag = visual.TextStim(win=win, name='p_box3_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    p_box3_P = visual.TextStim(win=win, name='p_box3_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    p_box4_mag = visual.TextStim(win=win, name='p_box4_mag',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    p_box4_P = visual.TextStim(win=win, name='p_box4_P',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.055, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    p_choice = keyboard.Keyboard(deviceName='p_choice')
    
    # --- Initialize components for Routine "gamble_choice" ---
    fixation_cross_3 = visual.ShapeStim(
        win=win, name='fixation_cross_3', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    gamble1_rew_test_2 = visual.TextStim(win=win, name='gamble1_rew_test_2',
        text='',
        font='Arial',
        pos=(-width, reward[2]+rew_txt_margin+0.03), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    gamble1_rew_text_low_2 = visual.TextStim(win=win, name='gamble1_rew_text_low_2',
        text='',
        font='Arial',
        pos=(-width, reward[2]+rew_txt_margin), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    gamble1_prob_text_2 = visual.TextStim(win=win, name='gamble1_prob_text_2',
        text='',
        font='Arial',
        pos=(-width, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    gamble2_rew_text_2 = visual.TextStim(win=win, name='gamble2_rew_text_2',
        text='',
        font='Arial',
        pos=(width, reward[2]+rew_txt_margin+0.03), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    gamble2_rew_text_low_2 = visual.TextStim(win=win, name='gamble2_rew_text_low_2',
        text='',
        font='Arial',
        pos=(width, reward[2]+rew_txt_margin), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    gamble2_prob_text_2 = visual.TextStim(win=win, name='gamble2_prob_text_2',
        text='',
        font='Arial',
        pos=(width, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    gamblechoice = keyboard.Keyboard(deviceName='gamblechoice')
    Choose_text = visual.TextStim(win=win, name='Choose_text',
        text='Press the corresponding arrow key to select a gamble.',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "chosen_gamble" ---
    fixation_7 = visual.ShapeStim(
        win=win, name='fixation_7', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, 1.0000, 1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    chosen_reward_txt = visual.TextStim(win=win, name='chosen_reward_txt',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    chosen_reward_txt_low = visual.TextStim(win=win, name='chosen_reward_txt_low',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    chosen_prob_txt = visual.TextStim(win=win, name='chosen_prob_txt',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "gamble_outcome" ---
    # Run 'Begin Experiment' code from gamble_result_code
    progVal = 1
    
    fixation_8 = visual.ShapeStim(
        win=win, name='fixation_8', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, 1.0000, 1.0000], fillColor=[1.0000, 1.0000, 1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    money_prompt = visual.TextStim(win=win, name='money_prompt',
        text='Current total: ',
        font='Arial',
        pos=(-0.55, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    current_money_txt = visual.TextStim(win=win, name='current_money_txt',
        text='',
        font='Arial',
        pos=(-0.3, 0.4), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color=[1.0000, 0.6863, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    outcome_square = visual.Rect(
        win=win, name='outcome_square',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    outcome_text_3 = visual.TextStim(win=win, name='outcome_text_3',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    outcome_text_2 = visual.TextStim(win=win, name='outcome_text_2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    prog_bar = visual.Progress(
        win, name='prog_bar',
        progress=0.0,
        pos=(-0.5, -0.45), size=(1, 0.03), anchor='center-left', units='height',
        barColor='white', backColor=None, borderColor=None, colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-7
    )
    next_trial_txt = visual.TextStim(win=win, name='next_trial_txt',
        text='Press the <space bar> to initiate the next trial.',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    next_trial_input = keyboard.Keyboard(deviceName='next_trial_input')
    
    # --- Initialize components for Routine "iti" ---
    text = visual.TextStim(win=win, name='text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "happiness_trial" ---
    happiness_rating = visual.Slider(win=win, name='happiness_rating',
        startValue=None, size=(1.3, 0.05), pos=(0, 0), units=win.units,
        labels=('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'), ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=0, readOnly=False)
    low_end_text_2 = visual.TextStim(win=win, name='low_end_text_2',
        text='very unhappy',
        font='Arial',
        pos=(-0.5, -0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    high_end_text_2 = visual.TextStim(win=win, name='high_end_text_2',
        text='very happy',
        font='Arial',
        pos=(0.5, -0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    question_2 = visual.TextStim(win=win, name='question_2',
        text='How happy are you at this moment?',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    exit_text = visual.TextStim(win=win, name='exit_text',
        text='Press <enter> to move to the next trial',
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_6 = keyboard.Keyboard(deviceName='key_resp_6')
    
    # --- Initialize components for Routine "h_iti" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial_break" ---
    break_txt = visual.TextStim(win=win, name='break_txt',
        text='Break\n\nPress <enter> when you are ready to resume',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_10 = keyboard.Keyboard(deviceName='key_resp_10')
    
    # --- Initialize components for Routine "repeat_iti_2" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Thank_you" ---
    thankYouText = visual.TextStim(win=win, name='thankYouText',
        text='You have completed the task. Thank you for your participation!\n\nPlease do not close your browser until the screen says to do so.\n\nThe page should automatically redirect you to Prolific once you exit the experiment. In case there is an error, please note the following completion code: C1D29EVY.\n\n(Press enter to exit the experiment)',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_11 = keyboard.Keyboard(deviceName='key_resp_11')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "consent" ---
    # create an object to store info about Routine consent
    consent = data.Routine(
        name='consent',
        components=[text_9, button],
    )
    consent.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button to account for continued clicks & clear times on/off
    button.reset()
    # store start times for consent
    consent.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    consent.tStart = globalClock.getTime(format='float')
    consent.status = STARTED
    thisExp.addData('consent.started', consent.tStart)
    consent.maxDuration = None
    # keep track of which components have finished
    consentComponents = consent.components
    for thisComponent in consent.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "consent" ---
    consent.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_9* updates
        
        # if text_9 is starting this frame...
        if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_9.started')
            # update status
            text_9.status = STARTED
            text_9.setAutoDraw(True)
        
        # if text_9 is active this frame...
        if text_9.status == STARTED:
            # update params
            pass
        # *button* updates
        
        # if button is starting this frame...
        if button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button.frameNStart = frameN  # exact frame index
            button.tStart = t  # local t and not account for scr refresh
            button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button.started')
            # update status
            button.status = STARTED
            win.callOnFlip(button.buttonClock.reset)
            button.setAutoDraw(True)
        
        # if button is active this frame...
        if button.status == STARTED:
            # update params
            pass
            # check whether button has been pressed
            if button.isClicked:
                if not button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button.timesOn.append(button.buttonClock.getTime())
                    button.timesOff.append(button.buttonClock.getTime())
                elif len(button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button.timesOff[-1] = button.buttonClock.getTime()
                if not button.wasClicked:
                    # end routine when button is clicked
                    continueRoutine = False
                if not button.wasClicked:
                    # run callback code when button is clicked
                    pass
        # take note of whether button was clicked, so that next frame we know if clicks are new
        button.wasClicked = button.isClicked and button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            consent.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in consent.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "consent" ---
    for thisComponent in consent.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for consent
    consent.tStop = globalClock.getTime(format='float')
    consent.tStopRefresh = tThisFlipGlobal
    thisExp.addData('consent.stopped', consent.tStop)
    thisExp.addData('button.numClicks', button.numClicks)
    if button.numClicks:
       thisExp.addData('button.timesOn', button.timesOn)
       thisExp.addData('button.timesOff', button.timesOff)
    else:
       thisExp.addData('button.timesOn', "")
       thisExp.addData('button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "consent" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Intros = data.TrialHandler2(
        name='Intros',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Introtext.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(Intros)  # add the loop to the experiment
    thisIntro = Intros.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIntro.rgb)
    if thisIntro != None:
        for paramName in thisIntro:
            globals()[paramName] = thisIntro[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisIntro in Intros:
        currentLoop = Intros
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisIntro.rgb)
        if thisIntro != None:
            for paramName in thisIntro:
                globals()[paramName] = thisIntro[paramName]
        
        # --- Prepare to start Routine "instructions1" ---
        # create an object to store info about Routine instructions1
        instructions1 = data.Routine(
            name='instructions1',
            components=[Intro_text_show, Continue_txt, key_resp],
        )
        instructions1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Intro_text_show.setText(Intro_text)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for instructions1
        instructions1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instructions1.tStart = globalClock.getTime(format='float')
        instructions1.status = STARTED
        thisExp.addData('instructions1.started', instructions1.tStart)
        instructions1.maxDuration = None
        # keep track of which components have finished
        instructions1Components = instructions1.components
        for thisComponent in instructions1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instructions1" ---
        # if trial has changed, end Routine now
        if isinstance(Intros, data.TrialHandler2) and thisIntro.thisN != Intros.thisTrial.thisN:
            continueRoutine = False
        instructions1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Intro_text_show* updates
            
            # if Intro_text_show is starting this frame...
            if Intro_text_show.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Intro_text_show.frameNStart = frameN  # exact frame index
                Intro_text_show.tStart = t  # local t and not account for scr refresh
                Intro_text_show.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Intro_text_show, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Intro_text_show.started')
                # update status
                Intro_text_show.status = STARTED
                Intro_text_show.setAutoDraw(True)
            
            # if Intro_text_show is active this frame...
            if Intro_text_show.status == STARTED:
                # update params
                pass
            
            # *Continue_txt* updates
            
            # if Continue_txt is starting this frame...
            if Continue_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Continue_txt.frameNStart = frameN  # exact frame index
                Continue_txt.tStart = t  # local t and not account for scr refresh
                Continue_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Continue_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Continue_txt.started')
                # update status
                Continue_txt.status = STARTED
                Continue_txt.setAutoDraw(True)
            
            # if Continue_txt is active this frame...
            if Continue_txt.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=True)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instructions1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instructions1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instructions1" ---
        for thisComponent in instructions1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instructions1
        instructions1.tStop = globalClock.getTime(format='float')
        instructions1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('instructions1.stopped', instructions1.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        Intros.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            Intros.addData('key_resp.rt', key_resp.rt)
            Intros.addData('key_resp.duration', key_resp.duration)
        # the Routine "instructions1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'Intros'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instructions_t1_target_on_choice" ---
    # create an object to store info about Routine instructions_t1_target_on_choice
    instructions_t1_target_on_choice = data.Routine(
        name='instructions_t1_target_on_choice',
        components=[fixation_inst_5a, intro_sure, intro_sure_txt, intro_gambletop, intro_gambletop_txt, intro_gambletop_p, intro_gamblelow, intro_gamblelow_txt, intro_gamblelow_p, text_inst_5a, choice_inst_5a],
    )
    instructions_t1_target_on_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_2
    Intro_sure = 0.5
    Intro_gamble_mag = 0.25
    Intro_gamble_p = 0.80
    intro_sure_txt.setText(Intro_sure)
    intro_gambletop_txt.setText(Intro_gamble_mag)
    intro_gambletop_p.setText([Intro_gamble_p + "%" ])
    # create starting attributes for choice_inst_5a
    choice_inst_5a.keys = []
    choice_inst_5a.rt = []
    _choice_inst_5a_allKeys = []
    # store start times for instructions_t1_target_on_choice
    instructions_t1_target_on_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_t1_target_on_choice.tStart = globalClock.getTime(format='float')
    instructions_t1_target_on_choice.status = STARTED
    thisExp.addData('instructions_t1_target_on_choice.started', instructions_t1_target_on_choice.tStart)
    instructions_t1_target_on_choice.maxDuration = None
    # keep track of which components have finished
    instructions_t1_target_on_choiceComponents = instructions_t1_target_on_choice.components
    for thisComponent in instructions_t1_target_on_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_t1_target_on_choice" ---
    instructions_t1_target_on_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_inst_5a* updates
        
        # if fixation_inst_5a is starting this frame...
        if fixation_inst_5a.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_inst_5a.frameNStart = frameN  # exact frame index
            fixation_inst_5a.tStart = t  # local t and not account for scr refresh
            fixation_inst_5a.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_inst_5a, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation_inst_5a.started')
            # update status
            fixation_inst_5a.status = STARTED
            fixation_inst_5a.setAutoDraw(True)
        
        # if fixation_inst_5a is active this frame...
        if fixation_inst_5a.status == STARTED:
            # update params
            pass
        
        # *intro_sure* updates
        
        # if intro_sure is starting this frame...
        if intro_sure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_sure.frameNStart = frameN  # exact frame index
            intro_sure.tStart = t  # local t and not account for scr refresh
            intro_sure.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_sure, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_sure.started')
            # update status
            intro_sure.status = STARTED
            intro_sure.setAutoDraw(True)
        
        # if intro_sure is active this frame...
        if intro_sure.status == STARTED:
            # update params
            pass
        
        # *intro_sure_txt* updates
        
        # if intro_sure_txt is starting this frame...
        if intro_sure_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_sure_txt.frameNStart = frameN  # exact frame index
            intro_sure_txt.tStart = t  # local t and not account for scr refresh
            intro_sure_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_sure_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_sure_txt.started')
            # update status
            intro_sure_txt.status = STARTED
            intro_sure_txt.setAutoDraw(True)
        
        # if intro_sure_txt is active this frame...
        if intro_sure_txt.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop* updates
        
        # if intro_gambletop is starting this frame...
        if intro_gambletop.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop.frameNStart = frameN  # exact frame index
            intro_gambletop.tStart = t  # local t and not account for scr refresh
            intro_gambletop.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop.started')
            # update status
            intro_gambletop.status = STARTED
            intro_gambletop.setAutoDraw(True)
        
        # if intro_gambletop is active this frame...
        if intro_gambletop.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop_txt* updates
        
        # if intro_gambletop_txt is starting this frame...
        if intro_gambletop_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop_txt.frameNStart = frameN  # exact frame index
            intro_gambletop_txt.tStart = t  # local t and not account for scr refresh
            intro_gambletop_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop_txt.started')
            # update status
            intro_gambletop_txt.status = STARTED
            intro_gambletop_txt.setAutoDraw(True)
        
        # if intro_gambletop_txt is active this frame...
        if intro_gambletop_txt.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop_p* updates
        
        # if intro_gambletop_p is starting this frame...
        if intro_gambletop_p.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop_p.frameNStart = frameN  # exact frame index
            intro_gambletop_p.tStart = t  # local t and not account for scr refresh
            intro_gambletop_p.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop_p, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop_p.started')
            # update status
            intro_gambletop_p.status = STARTED
            intro_gambletop_p.setAutoDraw(True)
        
        # if intro_gambletop_p is active this frame...
        if intro_gambletop_p.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow* updates
        
        # if intro_gamblelow is starting this frame...
        if intro_gamblelow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow.frameNStart = frameN  # exact frame index
            intro_gamblelow.tStart = t  # local t and not account for scr refresh
            intro_gamblelow.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow.started')
            # update status
            intro_gamblelow.status = STARTED
            intro_gamblelow.setAutoDraw(True)
        
        # if intro_gamblelow is active this frame...
        if intro_gamblelow.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow_txt* updates
        
        # if intro_gamblelow_txt is starting this frame...
        if intro_gamblelow_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow_txt.frameNStart = frameN  # exact frame index
            intro_gamblelow_txt.tStart = t  # local t and not account for scr refresh
            intro_gamblelow_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow_txt.started')
            # update status
            intro_gamblelow_txt.status = STARTED
            intro_gamblelow_txt.setAutoDraw(True)
        
        # if intro_gamblelow_txt is active this frame...
        if intro_gamblelow_txt.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow_p* updates
        
        # if intro_gamblelow_p is starting this frame...
        if intro_gamblelow_p.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow_p.frameNStart = frameN  # exact frame index
            intro_gamblelow_p.tStart = t  # local t and not account for scr refresh
            intro_gamblelow_p.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow_p, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow_p.started')
            # update status
            intro_gamblelow_p.status = STARTED
            intro_gamblelow_p.setAutoDraw(True)
        
        # if intro_gamblelow_p is active this frame...
        if intro_gamblelow_p.status == STARTED:
            # update params
            pass
        
        # *text_inst_5a* updates
        
        # if text_inst_5a is starting this frame...
        if text_inst_5a.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_inst_5a.frameNStart = frameN  # exact frame index
            text_inst_5a.tStart = t  # local t and not account for scr refresh
            text_inst_5a.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_inst_5a, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_inst_5a.started')
            # update status
            text_inst_5a.status = STARTED
            text_inst_5a.setAutoDraw(True)
        
        # if text_inst_5a is active this frame...
        if text_inst_5a.status == STARTED:
            # update params
            pass
        
        # *choice_inst_5a* updates
        waitOnFlip = False
        
        # if choice_inst_5a is starting this frame...
        if choice_inst_5a.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice_inst_5a.frameNStart = frameN  # exact frame index
            choice_inst_5a.tStart = t  # local t and not account for scr refresh
            choice_inst_5a.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice_inst_5a, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'choice_inst_5a.started')
            # update status
            choice_inst_5a.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(choice_inst_5a.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(choice_inst_5a.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if choice_inst_5a.status == STARTED and not waitOnFlip:
            theseKeys = choice_inst_5a.getKeys(keyList=['left'], ignoreKeys=["escape"], waitRelease=False)
            _choice_inst_5a_allKeys.extend(theseKeys)
            if len(_choice_inst_5a_allKeys):
                choice_inst_5a.keys = _choice_inst_5a_allKeys[-1].name  # just the last key pressed
                choice_inst_5a.rt = _choice_inst_5a_allKeys[-1].rt
                choice_inst_5a.duration = _choice_inst_5a_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_t1_target_on_choice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_t1_target_on_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_t1_target_on_choice" ---
    for thisComponent in instructions_t1_target_on_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_t1_target_on_choice
    instructions_t1_target_on_choice.tStop = globalClock.getTime(format='float')
    instructions_t1_target_on_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_t1_target_on_choice.stopped', instructions_t1_target_on_choice.tStop)
    # check responses
    if choice_inst_5a.keys in ['', [], None]:  # No response was made
        choice_inst_5a.keys = None
    thisExp.addData('choice_inst_5a.keys',choice_inst_5a.keys)
    if choice_inst_5a.keys != None:  # we had a response
        thisExp.addData('choice_inst_5a.rt', choice_inst_5a.rt)
        thisExp.addData('choice_inst_5a.duration', choice_inst_5a.duration)
    thisExp.nextEntry()
    # the Routine "instructions_t1_target_on_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_t1_chosen_gamble" ---
    # create an object to store info about Routine instructions_t1_chosen_gamble
    instructions_t1_chosen_gamble = data.Routine(
        name='instructions_t1_chosen_gamble',
        components=[fixation_inst_5a_2, intro_sure_2, intro_sure_txt_2],
    )
    instructions_t1_chosen_gamble.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    intro_sure_txt_2.setText(Intro_sure)
    # store start times for instructions_t1_chosen_gamble
    instructions_t1_chosen_gamble.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_t1_chosen_gamble.tStart = globalClock.getTime(format='float')
    instructions_t1_chosen_gamble.status = STARTED
    thisExp.addData('instructions_t1_chosen_gamble.started', instructions_t1_chosen_gamble.tStart)
    instructions_t1_chosen_gamble.maxDuration = time_chosen_gamble
    # keep track of which components have finished
    instructions_t1_chosen_gambleComponents = instructions_t1_chosen_gamble.components
    for thisComponent in instructions_t1_chosen_gamble.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_t1_chosen_gamble" ---
    instructions_t1_chosen_gamble.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > instructions_t1_chosen_gamble.maxDuration-frameTolerance:
            instructions_t1_chosen_gamble.maxDurationReached = True
            continueRoutine = False
        
        # *fixation_inst_5a_2* updates
        
        # if fixation_inst_5a_2 is starting this frame...
        if fixation_inst_5a_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_inst_5a_2.frameNStart = frameN  # exact frame index
            fixation_inst_5a_2.tStart = t  # local t and not account for scr refresh
            fixation_inst_5a_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_inst_5a_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation_inst_5a_2.started')
            # update status
            fixation_inst_5a_2.status = STARTED
            fixation_inst_5a_2.setAutoDraw(True)
        
        # if fixation_inst_5a_2 is active this frame...
        if fixation_inst_5a_2.status == STARTED:
            # update params
            pass
        
        # *intro_sure_2* updates
        
        # if intro_sure_2 is starting this frame...
        if intro_sure_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_sure_2.frameNStart = frameN  # exact frame index
            intro_sure_2.tStart = t  # local t and not account for scr refresh
            intro_sure_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_sure_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_sure_2.started')
            # update status
            intro_sure_2.status = STARTED
            intro_sure_2.setAutoDraw(True)
        
        # if intro_sure_2 is active this frame...
        if intro_sure_2.status == STARTED:
            # update params
            pass
        
        # *intro_sure_txt_2* updates
        
        # if intro_sure_txt_2 is starting this frame...
        if intro_sure_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_sure_txt_2.frameNStart = frameN  # exact frame index
            intro_sure_txt_2.tStart = t  # local t and not account for scr refresh
            intro_sure_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_sure_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_sure_txt_2.started')
            # update status
            intro_sure_txt_2.status = STARTED
            intro_sure_txt_2.setAutoDraw(True)
        
        # if intro_sure_txt_2 is active this frame...
        if intro_sure_txt_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_t1_chosen_gamble.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_t1_chosen_gamble.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_t1_chosen_gamble" ---
    for thisComponent in instructions_t1_chosen_gamble.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_t1_chosen_gamble
    instructions_t1_chosen_gamble.tStop = globalClock.getTime(format='float')
    instructions_t1_chosen_gamble.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_t1_chosen_gamble.stopped', instructions_t1_chosen_gamble.tStop)
    thisExp.nextEntry()
    # the Routine "instructions_t1_chosen_gamble" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_t1_outcome" ---
    # create an object to store info about Routine instructions_t1_outcome
    instructions_t1_outcome = data.Routine(
        name='instructions_t1_outcome',
        components=[total_prompt_inst_5a, curr_total_inst_5a, p_outcome_square_2, intro_outcome_text, intro_next_trial_txt_2, intro_next_trial_input_2],
    )
    instructions_t1_outcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    curr_total_inst_5a.setText('$5.5')
    p_outcome_square_2.setFillColor('green')
    p_outcome_square_2.setPos((0, 0))
    p_outcome_square_2.setSize((0.5*option_size*2, 0.5*option_size*2))
    p_outcome_square_2.setLineColor('green')
    # create starting attributes for intro_next_trial_input_2
    intro_next_trial_input_2.keys = []
    intro_next_trial_input_2.rt = []
    _intro_next_trial_input_2_allKeys = []
    # store start times for instructions_t1_outcome
    instructions_t1_outcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_t1_outcome.tStart = globalClock.getTime(format='float')
    instructions_t1_outcome.status = STARTED
    thisExp.addData('instructions_t1_outcome.started', instructions_t1_outcome.tStart)
    instructions_t1_outcome.maxDuration = None
    # keep track of which components have finished
    instructions_t1_outcomeComponents = instructions_t1_outcome.components
    for thisComponent in instructions_t1_outcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_t1_outcome" ---
    instructions_t1_outcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *total_prompt_inst_5a* updates
        
        # if total_prompt_inst_5a is starting this frame...
        if total_prompt_inst_5a.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            total_prompt_inst_5a.frameNStart = frameN  # exact frame index
            total_prompt_inst_5a.tStart = t  # local t and not account for scr refresh
            total_prompt_inst_5a.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(total_prompt_inst_5a, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'total_prompt_inst_5a.started')
            # update status
            total_prompt_inst_5a.status = STARTED
            total_prompt_inst_5a.setAutoDraw(True)
        
        # if total_prompt_inst_5a is active this frame...
        if total_prompt_inst_5a.status == STARTED:
            # update params
            pass
        
        # *curr_total_inst_5a* updates
        
        # if curr_total_inst_5a is starting this frame...
        if curr_total_inst_5a.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            curr_total_inst_5a.frameNStart = frameN  # exact frame index
            curr_total_inst_5a.tStart = t  # local t and not account for scr refresh
            curr_total_inst_5a.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(curr_total_inst_5a, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'curr_total_inst_5a.started')
            # update status
            curr_total_inst_5a.status = STARTED
            curr_total_inst_5a.setAutoDraw(True)
        
        # if curr_total_inst_5a is active this frame...
        if curr_total_inst_5a.status == STARTED:
            # update params
            pass
        
        # *p_outcome_square_2* updates
        
        # if p_outcome_square_2 is starting this frame...
        if p_outcome_square_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            p_outcome_square_2.frameNStart = frameN  # exact frame index
            p_outcome_square_2.tStart = t  # local t and not account for scr refresh
            p_outcome_square_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(p_outcome_square_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'p_outcome_square_2.started')
            # update status
            p_outcome_square_2.status = STARTED
            p_outcome_square_2.setAutoDraw(True)
        
        # if p_outcome_square_2 is active this frame...
        if p_outcome_square_2.status == STARTED:
            # update params
            pass
        
        # *intro_outcome_text* updates
        
        # if intro_outcome_text is starting this frame...
        if intro_outcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_outcome_text.frameNStart = frameN  # exact frame index
            intro_outcome_text.tStart = t  # local t and not account for scr refresh
            intro_outcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_outcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_outcome_text.started')
            # update status
            intro_outcome_text.status = STARTED
            intro_outcome_text.setAutoDraw(True)
        
        # if intro_outcome_text is active this frame...
        if intro_outcome_text.status == STARTED:
            # update params
            pass
        
        # *intro_next_trial_txt_2* updates
        
        # if intro_next_trial_txt_2 is starting this frame...
        if intro_next_trial_txt_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_next_trial_txt_2.frameNStart = frameN  # exact frame index
            intro_next_trial_txt_2.tStart = t  # local t and not account for scr refresh
            intro_next_trial_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_next_trial_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_next_trial_txt_2.started')
            # update status
            intro_next_trial_txt_2.status = STARTED
            intro_next_trial_txt_2.setAutoDraw(True)
        
        # if intro_next_trial_txt_2 is active this frame...
        if intro_next_trial_txt_2.status == STARTED:
            # update params
            pass
        
        # *intro_next_trial_input_2* updates
        waitOnFlip = False
        
        # if intro_next_trial_input_2 is starting this frame...
        if intro_next_trial_input_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intro_next_trial_input_2.frameNStart = frameN  # exact frame index
            intro_next_trial_input_2.tStart = t  # local t and not account for scr refresh
            intro_next_trial_input_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_next_trial_input_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_next_trial_input_2.started')
            # update status
            intro_next_trial_input_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_next_trial_input_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_next_trial_input_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_next_trial_input_2.status == STARTED and not waitOnFlip:
            theseKeys = intro_next_trial_input_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_next_trial_input_2_allKeys.extend(theseKeys)
            if len(_intro_next_trial_input_2_allKeys):
                intro_next_trial_input_2.keys = _intro_next_trial_input_2_allKeys[-1].name  # just the last key pressed
                intro_next_trial_input_2.rt = _intro_next_trial_input_2_allKeys[-1].rt
                intro_next_trial_input_2.duration = _intro_next_trial_input_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_t1_outcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_t1_outcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_t1_outcome" ---
    for thisComponent in instructions_t1_outcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_t1_outcome
    instructions_t1_outcome.tStop = globalClock.getTime(format='float')
    instructions_t1_outcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_t1_outcome.stopped', instructions_t1_outcome.tStop)
    # check responses
    if intro_next_trial_input_2.keys in ['', [], None]:  # No response was made
        intro_next_trial_input_2.keys = None
    thisExp.addData('intro_next_trial_input_2.keys',intro_next_trial_input_2.keys)
    if intro_next_trial_input_2.keys != None:  # we had a response
        thisExp.addData('intro_next_trial_input_2.rt', intro_next_trial_input_2.rt)
        thisExp.addData('intro_next_trial_input_2.duration', intro_next_trial_input_2.duration)
    thisExp.nextEntry()
    # the Routine "instructions_t1_outcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_iti" ---
    # create an object to store info about Routine intro_iti
    intro_iti = data.Routine(
        name='intro_iti',
        components=[text_4],
    )
    intro_iti.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for intro_iti
    intro_iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro_iti.tStart = globalClock.getTime(format='float')
    intro_iti.status = STARTED
    thisExp.addData('intro_iti.started', intro_iti.tStart)
    intro_iti.maxDuration = time_iti
    # keep track of which components have finished
    intro_itiComponents = intro_iti.components
    for thisComponent in intro_iti.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_iti" ---
    intro_iti.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > intro_iti.maxDuration-frameTolerance:
            intro_iti.maxDurationReached = True
            continueRoutine = False
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            intro_iti.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_iti.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_iti" ---
    for thisComponent in intro_iti.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro_iti
    intro_iti.tStop = globalClock.getTime(format='float')
    intro_iti.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro_iti.stopped', intro_iti.tStop)
    thisExp.nextEntry()
    # the Routine "intro_iti" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_t2_target_on_choice" ---
    # create an object to store info about Routine instructions_t2_target_on_choice
    instructions_t2_target_on_choice = data.Routine(
        name='instructions_t2_target_on_choice',
        components=[fixation_inst_5b, intro_sure_3, intro_gambletop_2, intro_gamblelow_2, intro_sure_txt_3, intro_gambletop_txt_2, intro_gambletop_p_2, intro_gamblelow_txt_2, intro_gamblelow_p_2, text_inst_5b, choice_inst_5b],
    )
    instructions_t2_target_on_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    intro_sure_txt_3.setText('$0.15')
    intro_gambletop_txt_2.setText('$0.50')
    intro_gambletop_p_2.setText([0.80 + "%" ])
    # create starting attributes for choice_inst_5b
    choice_inst_5b.keys = []
    choice_inst_5b.rt = []
    _choice_inst_5b_allKeys = []
    # store start times for instructions_t2_target_on_choice
    instructions_t2_target_on_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_t2_target_on_choice.tStart = globalClock.getTime(format='float')
    instructions_t2_target_on_choice.status = STARTED
    thisExp.addData('instructions_t2_target_on_choice.started', instructions_t2_target_on_choice.tStart)
    instructions_t2_target_on_choice.maxDuration = None
    # keep track of which components have finished
    instructions_t2_target_on_choiceComponents = instructions_t2_target_on_choice.components
    for thisComponent in instructions_t2_target_on_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_t2_target_on_choice" ---
    instructions_t2_target_on_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_inst_5b* updates
        
        # if fixation_inst_5b is starting this frame...
        if fixation_inst_5b.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_inst_5b.frameNStart = frameN  # exact frame index
            fixation_inst_5b.tStart = t  # local t and not account for scr refresh
            fixation_inst_5b.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_inst_5b, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation_inst_5b.started')
            # update status
            fixation_inst_5b.status = STARTED
            fixation_inst_5b.setAutoDraw(True)
        
        # if fixation_inst_5b is active this frame...
        if fixation_inst_5b.status == STARTED:
            # update params
            pass
        
        # *intro_sure_3* updates
        
        # if intro_sure_3 is starting this frame...
        if intro_sure_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_sure_3.frameNStart = frameN  # exact frame index
            intro_sure_3.tStart = t  # local t and not account for scr refresh
            intro_sure_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_sure_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_sure_3.started')
            # update status
            intro_sure_3.status = STARTED
            intro_sure_3.setAutoDraw(True)
        
        # if intro_sure_3 is active this frame...
        if intro_sure_3.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop_2* updates
        
        # if intro_gambletop_2 is starting this frame...
        if intro_gambletop_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop_2.frameNStart = frameN  # exact frame index
            intro_gambletop_2.tStart = t  # local t and not account for scr refresh
            intro_gambletop_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop_2.started')
            # update status
            intro_gambletop_2.status = STARTED
            intro_gambletop_2.setAutoDraw(True)
        
        # if intro_gambletop_2 is active this frame...
        if intro_gambletop_2.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow_2* updates
        
        # if intro_gamblelow_2 is starting this frame...
        if intro_gamblelow_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow_2.frameNStart = frameN  # exact frame index
            intro_gamblelow_2.tStart = t  # local t and not account for scr refresh
            intro_gamblelow_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow_2.started')
            # update status
            intro_gamblelow_2.status = STARTED
            intro_gamblelow_2.setAutoDraw(True)
        
        # if intro_gamblelow_2 is active this frame...
        if intro_gamblelow_2.status == STARTED:
            # update params
            pass
        
        # *intro_sure_txt_3* updates
        
        # if intro_sure_txt_3 is starting this frame...
        if intro_sure_txt_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_sure_txt_3.frameNStart = frameN  # exact frame index
            intro_sure_txt_3.tStart = t  # local t and not account for scr refresh
            intro_sure_txt_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_sure_txt_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_sure_txt_3.started')
            # update status
            intro_sure_txt_3.status = STARTED
            intro_sure_txt_3.setAutoDraw(True)
        
        # if intro_sure_txt_3 is active this frame...
        if intro_sure_txt_3.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop_txt_2* updates
        
        # if intro_gambletop_txt_2 is starting this frame...
        if intro_gambletop_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop_txt_2.frameNStart = frameN  # exact frame index
            intro_gambletop_txt_2.tStart = t  # local t and not account for scr refresh
            intro_gambletop_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop_txt_2.started')
            # update status
            intro_gambletop_txt_2.status = STARTED
            intro_gambletop_txt_2.setAutoDraw(True)
        
        # if intro_gambletop_txt_2 is active this frame...
        if intro_gambletop_txt_2.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop_p_2* updates
        
        # if intro_gambletop_p_2 is starting this frame...
        if intro_gambletop_p_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop_p_2.frameNStart = frameN  # exact frame index
            intro_gambletop_p_2.tStart = t  # local t and not account for scr refresh
            intro_gambletop_p_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop_p_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop_p_2.started')
            # update status
            intro_gambletop_p_2.status = STARTED
            intro_gambletop_p_2.setAutoDraw(True)
        
        # if intro_gambletop_p_2 is active this frame...
        if intro_gambletop_p_2.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow_txt_2* updates
        
        # if intro_gamblelow_txt_2 is starting this frame...
        if intro_gamblelow_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow_txt_2.frameNStart = frameN  # exact frame index
            intro_gamblelow_txt_2.tStart = t  # local t and not account for scr refresh
            intro_gamblelow_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow_txt_2.started')
            # update status
            intro_gamblelow_txt_2.status = STARTED
            intro_gamblelow_txt_2.setAutoDraw(True)
        
        # if intro_gamblelow_txt_2 is active this frame...
        if intro_gamblelow_txt_2.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow_p_2* updates
        
        # if intro_gamblelow_p_2 is starting this frame...
        if intro_gamblelow_p_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow_p_2.frameNStart = frameN  # exact frame index
            intro_gamblelow_p_2.tStart = t  # local t and not account for scr refresh
            intro_gamblelow_p_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow_p_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow_p_2.started')
            # update status
            intro_gamblelow_p_2.status = STARTED
            intro_gamblelow_p_2.setAutoDraw(True)
        
        # if intro_gamblelow_p_2 is active this frame...
        if intro_gamblelow_p_2.status == STARTED:
            # update params
            pass
        
        # *text_inst_5b* updates
        
        # if text_inst_5b is starting this frame...
        if text_inst_5b.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_inst_5b.frameNStart = frameN  # exact frame index
            text_inst_5b.tStart = t  # local t and not account for scr refresh
            text_inst_5b.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_inst_5b, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_inst_5b.started')
            # update status
            text_inst_5b.status = STARTED
            text_inst_5b.setAutoDraw(True)
        
        # if text_inst_5b is active this frame...
        if text_inst_5b.status == STARTED:
            # update params
            pass
        
        # *choice_inst_5b* updates
        waitOnFlip = False
        
        # if choice_inst_5b is starting this frame...
        if choice_inst_5b.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice_inst_5b.frameNStart = frameN  # exact frame index
            choice_inst_5b.tStart = t  # local t and not account for scr refresh
            choice_inst_5b.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice_inst_5b, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'choice_inst_5b.started')
            # update status
            choice_inst_5b.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(choice_inst_5b.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(choice_inst_5b.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if choice_inst_5b.status == STARTED and not waitOnFlip:
            theseKeys = choice_inst_5b.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
            _choice_inst_5b_allKeys.extend(theseKeys)
            if len(_choice_inst_5b_allKeys):
                choice_inst_5b.keys = _choice_inst_5b_allKeys[-1].name  # just the last key pressed
                choice_inst_5b.rt = _choice_inst_5b_allKeys[-1].rt
                choice_inst_5b.duration = _choice_inst_5b_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_t2_target_on_choice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_t2_target_on_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_t2_target_on_choice" ---
    for thisComponent in instructions_t2_target_on_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_t2_target_on_choice
    instructions_t2_target_on_choice.tStop = globalClock.getTime(format='float')
    instructions_t2_target_on_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_t2_target_on_choice.stopped', instructions_t2_target_on_choice.tStop)
    # check responses
    if choice_inst_5b.keys in ['', [], None]:  # No response was made
        choice_inst_5b.keys = None
    thisExp.addData('choice_inst_5b.keys',choice_inst_5b.keys)
    if choice_inst_5b.keys != None:  # we had a response
        thisExp.addData('choice_inst_5b.rt', choice_inst_5b.rt)
        thisExp.addData('choice_inst_5b.duration', choice_inst_5b.duration)
    thisExp.nextEntry()
    # the Routine "instructions_t2_target_on_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_t2_chosen_gamble" ---
    # create an object to store info about Routine instructions_t2_chosen_gamble
    instructions_t2_chosen_gamble = data.Routine(
        name='instructions_t2_chosen_gamble',
        components=[fixation_inst_5a_3, intro_gambletop_3, intro_gamblelow_3, intro_gambletop_txt_3, intro_gambletop_p_3, intro_gamblelow_txt_3, intro_gamblelow_p_3],
    )
    instructions_t2_chosen_gamble.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    intro_gambletop_txt_3.setText('$0.50')
    intro_gambletop_p_3.setText([0.80 + "%" ])
    # store start times for instructions_t2_chosen_gamble
    instructions_t2_chosen_gamble.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_t2_chosen_gamble.tStart = globalClock.getTime(format='float')
    instructions_t2_chosen_gamble.status = STARTED
    thisExp.addData('instructions_t2_chosen_gamble.started', instructions_t2_chosen_gamble.tStart)
    instructions_t2_chosen_gamble.maxDuration = time_chosen_gamble
    # keep track of which components have finished
    instructions_t2_chosen_gambleComponents = instructions_t2_chosen_gamble.components
    for thisComponent in instructions_t2_chosen_gamble.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_t2_chosen_gamble" ---
    instructions_t2_chosen_gamble.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > instructions_t2_chosen_gamble.maxDuration-frameTolerance:
            instructions_t2_chosen_gamble.maxDurationReached = True
            continueRoutine = False
        
        # *fixation_inst_5a_3* updates
        
        # if fixation_inst_5a_3 is starting this frame...
        if fixation_inst_5a_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_inst_5a_3.frameNStart = frameN  # exact frame index
            fixation_inst_5a_3.tStart = t  # local t and not account for scr refresh
            fixation_inst_5a_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_inst_5a_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation_inst_5a_3.started')
            # update status
            fixation_inst_5a_3.status = STARTED
            fixation_inst_5a_3.setAutoDraw(True)
        
        # if fixation_inst_5a_3 is active this frame...
        if fixation_inst_5a_3.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop_3* updates
        
        # if intro_gambletop_3 is starting this frame...
        if intro_gambletop_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop_3.frameNStart = frameN  # exact frame index
            intro_gambletop_3.tStart = t  # local t and not account for scr refresh
            intro_gambletop_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop_3.started')
            # update status
            intro_gambletop_3.status = STARTED
            intro_gambletop_3.setAutoDraw(True)
        
        # if intro_gambletop_3 is active this frame...
        if intro_gambletop_3.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow_3* updates
        
        # if intro_gamblelow_3 is starting this frame...
        if intro_gamblelow_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow_3.frameNStart = frameN  # exact frame index
            intro_gamblelow_3.tStart = t  # local t and not account for scr refresh
            intro_gamblelow_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow_3.started')
            # update status
            intro_gamblelow_3.status = STARTED
            intro_gamblelow_3.setAutoDraw(True)
        
        # if intro_gamblelow_3 is active this frame...
        if intro_gamblelow_3.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop_txt_3* updates
        
        # if intro_gambletop_txt_3 is starting this frame...
        if intro_gambletop_txt_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop_txt_3.frameNStart = frameN  # exact frame index
            intro_gambletop_txt_3.tStart = t  # local t and not account for scr refresh
            intro_gambletop_txt_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop_txt_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop_txt_3.started')
            # update status
            intro_gambletop_txt_3.status = STARTED
            intro_gambletop_txt_3.setAutoDraw(True)
        
        # if intro_gambletop_txt_3 is active this frame...
        if intro_gambletop_txt_3.status == STARTED:
            # update params
            pass
        
        # *intro_gambletop_p_3* updates
        
        # if intro_gambletop_p_3 is starting this frame...
        if intro_gambletop_p_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gambletop_p_3.frameNStart = frameN  # exact frame index
            intro_gambletop_p_3.tStart = t  # local t and not account for scr refresh
            intro_gambletop_p_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gambletop_p_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gambletop_p_3.started')
            # update status
            intro_gambletop_p_3.status = STARTED
            intro_gambletop_p_3.setAutoDraw(True)
        
        # if intro_gambletop_p_3 is active this frame...
        if intro_gambletop_p_3.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow_txt_3* updates
        
        # if intro_gamblelow_txt_3 is starting this frame...
        if intro_gamblelow_txt_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow_txt_3.frameNStart = frameN  # exact frame index
            intro_gamblelow_txt_3.tStart = t  # local t and not account for scr refresh
            intro_gamblelow_txt_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow_txt_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow_txt_3.started')
            # update status
            intro_gamblelow_txt_3.status = STARTED
            intro_gamblelow_txt_3.setAutoDraw(True)
        
        # if intro_gamblelow_txt_3 is active this frame...
        if intro_gamblelow_txt_3.status == STARTED:
            # update params
            pass
        
        # *intro_gamblelow_p_3* updates
        
        # if intro_gamblelow_p_3 is starting this frame...
        if intro_gamblelow_p_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_gamblelow_p_3.frameNStart = frameN  # exact frame index
            intro_gamblelow_p_3.tStart = t  # local t and not account for scr refresh
            intro_gamblelow_p_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_gamblelow_p_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_gamblelow_p_3.started')
            # update status
            intro_gamblelow_p_3.status = STARTED
            intro_gamblelow_p_3.setAutoDraw(True)
        
        # if intro_gamblelow_p_3 is active this frame...
        if intro_gamblelow_p_3.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_t2_chosen_gamble.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_t2_chosen_gamble.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_t2_chosen_gamble" ---
    for thisComponent in instructions_t2_chosen_gamble.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_t2_chosen_gamble
    instructions_t2_chosen_gamble.tStop = globalClock.getTime(format='float')
    instructions_t2_chosen_gamble.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_t2_chosen_gamble.stopped', instructions_t2_chosen_gamble.tStop)
    thisExp.nextEntry()
    # the Routine "instructions_t2_chosen_gamble" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_t2_outcome" ---
    # create an object to store info about Routine instructions_t2_outcome
    instructions_t2_outcome = data.Routine(
        name='instructions_t2_outcome',
        components=[total_prompt_inst_5a_2, curr_total_inst_5a_2, p_outcome_square_3, intro_outcome_text_3, intro_next_trial_txt, intro_next_trial_input],
    )
    instructions_t2_outcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    curr_total_inst_5a_2.setText('$5.50')
    p_outcome_square_3.setFillColor('red')
    p_outcome_square_3.setPos((0, 0))
    p_outcome_square_3.setSize((0.4*option_size*2, 0.4*option_size*2))
    p_outcome_square_3.setLineColor('red')
    # create starting attributes for intro_next_trial_input
    intro_next_trial_input.keys = []
    intro_next_trial_input.rt = []
    _intro_next_trial_input_allKeys = []
    # store start times for instructions_t2_outcome
    instructions_t2_outcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_t2_outcome.tStart = globalClock.getTime(format='float')
    instructions_t2_outcome.status = STARTED
    thisExp.addData('instructions_t2_outcome.started', instructions_t2_outcome.tStart)
    instructions_t2_outcome.maxDuration = None
    # keep track of which components have finished
    instructions_t2_outcomeComponents = instructions_t2_outcome.components
    for thisComponent in instructions_t2_outcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_t2_outcome" ---
    instructions_t2_outcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *total_prompt_inst_5a_2* updates
        
        # if total_prompt_inst_5a_2 is starting this frame...
        if total_prompt_inst_5a_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            total_prompt_inst_5a_2.frameNStart = frameN  # exact frame index
            total_prompt_inst_5a_2.tStart = t  # local t and not account for scr refresh
            total_prompt_inst_5a_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(total_prompt_inst_5a_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'total_prompt_inst_5a_2.started')
            # update status
            total_prompt_inst_5a_2.status = STARTED
            total_prompt_inst_5a_2.setAutoDraw(True)
        
        # if total_prompt_inst_5a_2 is active this frame...
        if total_prompt_inst_5a_2.status == STARTED:
            # update params
            pass
        
        # *curr_total_inst_5a_2* updates
        
        # if curr_total_inst_5a_2 is starting this frame...
        if curr_total_inst_5a_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            curr_total_inst_5a_2.frameNStart = frameN  # exact frame index
            curr_total_inst_5a_2.tStart = t  # local t and not account for scr refresh
            curr_total_inst_5a_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(curr_total_inst_5a_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'curr_total_inst_5a_2.started')
            # update status
            curr_total_inst_5a_2.status = STARTED
            curr_total_inst_5a_2.setAutoDraw(True)
        
        # if curr_total_inst_5a_2 is active this frame...
        if curr_total_inst_5a_2.status == STARTED:
            # update params
            pass
        
        # *p_outcome_square_3* updates
        
        # if p_outcome_square_3 is starting this frame...
        if p_outcome_square_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            p_outcome_square_3.frameNStart = frameN  # exact frame index
            p_outcome_square_3.tStart = t  # local t and not account for scr refresh
            p_outcome_square_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(p_outcome_square_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'p_outcome_square_3.started')
            # update status
            p_outcome_square_3.status = STARTED
            p_outcome_square_3.setAutoDraw(True)
        
        # if p_outcome_square_3 is active this frame...
        if p_outcome_square_3.status == STARTED:
            # update params
            pass
        
        # *intro_outcome_text_3* updates
        
        # if intro_outcome_text_3 is starting this frame...
        if intro_outcome_text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_outcome_text_3.frameNStart = frameN  # exact frame index
            intro_outcome_text_3.tStart = t  # local t and not account for scr refresh
            intro_outcome_text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_outcome_text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_outcome_text_3.started')
            # update status
            intro_outcome_text_3.status = STARTED
            intro_outcome_text_3.setAutoDraw(True)
        
        # if intro_outcome_text_3 is active this frame...
        if intro_outcome_text_3.status == STARTED:
            # update params
            pass
        
        # *intro_next_trial_txt* updates
        
        # if intro_next_trial_txt is starting this frame...
        if intro_next_trial_txt.status == NOT_STARTED and tThisFlip >= time_gamble_result-frameTolerance:
            # keep track of start time/frame for later
            intro_next_trial_txt.frameNStart = frameN  # exact frame index
            intro_next_trial_txt.tStart = t  # local t and not account for scr refresh
            intro_next_trial_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_next_trial_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_next_trial_txt.started')
            # update status
            intro_next_trial_txt.status = STARTED
            intro_next_trial_txt.setAutoDraw(True)
        
        # if intro_next_trial_txt is active this frame...
        if intro_next_trial_txt.status == STARTED:
            # update params
            pass
        
        # *intro_next_trial_input* updates
        waitOnFlip = False
        
        # if intro_next_trial_input is starting this frame...
        if intro_next_trial_input.status == NOT_STARTED and tThisFlip >= time_gamble_result-frameTolerance:
            # keep track of start time/frame for later
            intro_next_trial_input.frameNStart = frameN  # exact frame index
            intro_next_trial_input.tStart = t  # local t and not account for scr refresh
            intro_next_trial_input.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_next_trial_input, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_next_trial_input.started')
            # update status
            intro_next_trial_input.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_next_trial_input.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_next_trial_input.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_next_trial_input.status == STARTED and not waitOnFlip:
            theseKeys = intro_next_trial_input.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_next_trial_input_allKeys.extend(theseKeys)
            if len(_intro_next_trial_input_allKeys):
                intro_next_trial_input.keys = _intro_next_trial_input_allKeys[-1].name  # just the last key pressed
                intro_next_trial_input.rt = _intro_next_trial_input_allKeys[-1].rt
                intro_next_trial_input.duration = _intro_next_trial_input_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_t2_outcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_t2_outcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_t2_outcome" ---
    for thisComponent in instructions_t2_outcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_t2_outcome
    instructions_t2_outcome.tStop = globalClock.getTime(format='float')
    instructions_t2_outcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_t2_outcome.stopped', instructions_t2_outcome.tStop)
    # check responses
    if intro_next_trial_input.keys in ['', [], None]:  # No response was made
        intro_next_trial_input.keys = None
    thisExp.addData('intro_next_trial_input.keys',intro_next_trial_input.keys)
    if intro_next_trial_input.keys != None:  # we had a response
        thisExp.addData('intro_next_trial_input.rt', intro_next_trial_input.rt)
        thisExp.addData('intro_next_trial_input.duration', intro_next_trial_input.duration)
    thisExp.nextEntry()
    # the Routine "instructions_t2_outcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions6" ---
    # create an object to store info about Routine instructions6
    instructions6 = data.Routine(
        name='instructions6',
        components=[instructions_text, Continue_txt_2, key_resp_inst_5],
    )
    instructions6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_inst_5
    key_resp_inst_5.keys = []
    key_resp_inst_5.rt = []
    _key_resp_inst_5_allKeys = []
    # store start times for instructions6
    instructions6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions6.tStart = globalClock.getTime(format='float')
    instructions6.status = STARTED
    thisExp.addData('instructions6.started', instructions6.tStart)
    instructions6.maxDuration = None
    # keep track of which components have finished
    instructions6Components = instructions6.components
    for thisComponent in instructions6.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions6" ---
    instructions6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_text* updates
        
        # if instructions_text is starting this frame...
        if instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_text.frameNStart = frameN  # exact frame index
            instructions_text.tStart = t  # local t and not account for scr refresh
            instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_text.started')
            # update status
            instructions_text.status = STARTED
            instructions_text.setAutoDraw(True)
        
        # if instructions_text is active this frame...
        if instructions_text.status == STARTED:
            # update params
            pass
        
        # *Continue_txt_2* updates
        
        # if Continue_txt_2 is starting this frame...
        if Continue_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Continue_txt_2.frameNStart = frameN  # exact frame index
            Continue_txt_2.tStart = t  # local t and not account for scr refresh
            Continue_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Continue_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Continue_txt_2.started')
            # update status
            Continue_txt_2.status = STARTED
            Continue_txt_2.setAutoDraw(True)
        
        # if Continue_txt_2 is active this frame...
        if Continue_txt_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst_5* updates
        waitOnFlip = False
        
        # if key_resp_inst_5 is starting this frame...
        if key_resp_inst_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst_5.frameNStart = frameN  # exact frame index
            key_resp_inst_5.tStart = t  # local t and not account for scr refresh
            key_resp_inst_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_inst_5.started')
            # update status
            key_resp_inst_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst_5.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_5_allKeys.extend(theseKeys)
            if len(_key_resp_inst_5_allKeys):
                key_resp_inst_5.keys = _key_resp_inst_5_allKeys[-1].name  # just the last key pressed
                key_resp_inst_5.rt = _key_resp_inst_5_allKeys[-1].rt
                key_resp_inst_5.duration = _key_resp_inst_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions6.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions6" ---
    for thisComponent in instructions6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions6
    instructions6.tStop = globalClock.getTime(format='float')
    instructions6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions6.stopped', instructions6.tStop)
    # check responses
    if key_resp_inst_5.keys in ['', [], None]:  # No response was made
        key_resp_inst_5.keys = None
    thisExp.addData('key_resp_inst_5.keys',key_resp_inst_5.keys)
    if key_resp_inst_5.keys != None:  # we had a response
        thisExp.addData('key_resp_inst_5.rt', key_resp_inst_5.rt)
        thisExp.addData('key_resp_inst_5.duration', key_resp_inst_5.duration)
    thisExp.nextEntry()
    # the Routine "instructions6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions7" ---
    # create an object to store info about Routine instructions7
    instructions7 = data.Routine(
        name='instructions7',
        components=[fixation_inst_5a_6, total_prompt_inst, curr_total_inst, p_outcome_square_4, intro_outcome_text_5, key_resp_inst_6, instrutions_text, prog_5, arrow1, arrow2],
    )
    instructions7.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    p_outcome_square_4.setFillColor('red')
    p_outcome_square_4.setPos((-width, 0))
    p_outcome_square_4.setSize((reward[1]*pie_size*2, reward[1]*pie_size*2))
    p_outcome_square_4.setLineColor('red')
    # create starting attributes for key_resp_inst_6
    key_resp_inst_6.keys = []
    key_resp_inst_6.rt = []
    _key_resp_inst_6_allKeys = []
    # store start times for instructions7
    instructions7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions7.tStart = globalClock.getTime(format='float')
    instructions7.status = STARTED
    thisExp.addData('instructions7.started', instructions7.tStart)
    instructions7.maxDuration = None
    # keep track of which components have finished
    instructions7Components = instructions7.components
    for thisComponent in instructions7.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions7" ---
    instructions7.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_inst_5a_6* updates
        
        # if fixation_inst_5a_6 is starting this frame...
        if fixation_inst_5a_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_inst_5a_6.frameNStart = frameN  # exact frame index
            fixation_inst_5a_6.tStart = t  # local t and not account for scr refresh
            fixation_inst_5a_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_inst_5a_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation_inst_5a_6.started')
            # update status
            fixation_inst_5a_6.status = STARTED
            fixation_inst_5a_6.setAutoDraw(True)
        
        # if fixation_inst_5a_6 is active this frame...
        if fixation_inst_5a_6.status == STARTED:
            # update params
            pass
        
        # *total_prompt_inst* updates
        
        # if total_prompt_inst is starting this frame...
        if total_prompt_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            total_prompt_inst.frameNStart = frameN  # exact frame index
            total_prompt_inst.tStart = t  # local t and not account for scr refresh
            total_prompt_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(total_prompt_inst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'total_prompt_inst.started')
            # update status
            total_prompt_inst.status = STARTED
            total_prompt_inst.setAutoDraw(True)
        
        # if total_prompt_inst is active this frame...
        if total_prompt_inst.status == STARTED:
            # update params
            pass
        
        # *curr_total_inst* updates
        
        # if curr_total_inst is starting this frame...
        if curr_total_inst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            curr_total_inst.frameNStart = frameN  # exact frame index
            curr_total_inst.tStart = t  # local t and not account for scr refresh
            curr_total_inst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(curr_total_inst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'curr_total_inst.started')
            # update status
            curr_total_inst.status = STARTED
            curr_total_inst.setAutoDraw(True)
        
        # if curr_total_inst is active this frame...
        if curr_total_inst.status == STARTED:
            # update params
            pass
        
        # *p_outcome_square_4* updates
        
        # if p_outcome_square_4 is starting this frame...
        if p_outcome_square_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            p_outcome_square_4.frameNStart = frameN  # exact frame index
            p_outcome_square_4.tStart = t  # local t and not account for scr refresh
            p_outcome_square_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(p_outcome_square_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'p_outcome_square_4.started')
            # update status
            p_outcome_square_4.status = STARTED
            p_outcome_square_4.setAutoDraw(True)
        
        # if p_outcome_square_4 is active this frame...
        if p_outcome_square_4.status == STARTED:
            # update params
            pass
        
        # *intro_outcome_text_5* updates
        
        # if intro_outcome_text_5 is starting this frame...
        if intro_outcome_text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_outcome_text_5.frameNStart = frameN  # exact frame index
            intro_outcome_text_5.tStart = t  # local t and not account for scr refresh
            intro_outcome_text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_outcome_text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_outcome_text_5.started')
            # update status
            intro_outcome_text_5.status = STARTED
            intro_outcome_text_5.setAutoDraw(True)
        
        # if intro_outcome_text_5 is active this frame...
        if intro_outcome_text_5.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst_6* updates
        waitOnFlip = False
        
        # if key_resp_inst_6 is starting this frame...
        if key_resp_inst_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst_6.frameNStart = frameN  # exact frame index
            key_resp_inst_6.tStart = t  # local t and not account for scr refresh
            key_resp_inst_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_inst_6.started')
            # update status
            key_resp_inst_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst_6.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst_6.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_6_allKeys.extend(theseKeys)
            if len(_key_resp_inst_6_allKeys):
                key_resp_inst_6.keys = _key_resp_inst_6_allKeys[-1].name  # just the last key pressed
                key_resp_inst_6.rt = _key_resp_inst_6_allKeys[-1].rt
                key_resp_inst_6.duration = _key_resp_inst_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instrutions_text* updates
        
        # if instrutions_text is starting this frame...
        if instrutions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instrutions_text.frameNStart = frameN  # exact frame index
            instrutions_text.tStart = t  # local t and not account for scr refresh
            instrutions_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instrutions_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instrutions_text.started')
            # update status
            instrutions_text.status = STARTED
            instrutions_text.setAutoDraw(True)
        
        # if instrutions_text is active this frame...
        if instrutions_text.status == STARTED:
            # update params
            pass
        
        # *prog_5* updates
        
        # if prog_5 is starting this frame...
        if prog_5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            prog_5.frameNStart = frameN  # exact frame index
            prog_5.tStart = t  # local t and not account for scr refresh
            prog_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prog_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prog_5.started')
            # update status
            prog_5.status = STARTED
            prog_5.setAutoDraw(True)
        
        # if prog_5 is active this frame...
        if prog_5.status == STARTED:
            # update params
            pass
        
        # *arrow1* updates
        
        # if arrow1 is starting this frame...
        if arrow1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            arrow1.frameNStart = frameN  # exact frame index
            arrow1.tStart = t  # local t and not account for scr refresh
            arrow1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(arrow1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'arrow1.started')
            # update status
            arrow1.status = STARTED
            arrow1.setAutoDraw(True)
        
        # if arrow1 is active this frame...
        if arrow1.status == STARTED:
            # update params
            pass
        
        # *arrow2* updates
        
        # if arrow2 is starting this frame...
        if arrow2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            arrow2.frameNStart = frameN  # exact frame index
            arrow2.tStart = t  # local t and not account for scr refresh
            arrow2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(arrow2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'arrow2.started')
            # update status
            arrow2.status = STARTED
            arrow2.setAutoDraw(True)
        
        # if arrow2 is active this frame...
        if arrow2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions7.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions7.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions7" ---
    for thisComponent in instructions7.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions7
    instructions7.tStop = globalClock.getTime(format='float')
    instructions7.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions7.stopped', instructions7.tStop)
    # check responses
    if key_resp_inst_6.keys in ['', [], None]:  # No response was made
        key_resp_inst_6.keys = None
    thisExp.addData('key_resp_inst_6.keys',key_resp_inst_6.keys)
    if key_resp_inst_6.keys != None:  # we had a response
        thisExp.addData('key_resp_inst_6.rt', key_resp_inst_6.rt)
        thisExp.addData('key_resp_inst_6.duration', key_resp_inst_6.duration)
    thisExp.nextEntry()
    # the Routine "instructions7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions8" ---
    # create an object to store info about Routine instructions8
    instructions8 = data.Routine(
        name='instructions8',
        components=[instructions_text_3, Continue_txt_3, key_resp_inst_7],
    )
    instructions8.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_inst_7
    key_resp_inst_7.keys = []
    key_resp_inst_7.rt = []
    _key_resp_inst_7_allKeys = []
    # store start times for instructions8
    instructions8.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions8.tStart = globalClock.getTime(format='float')
    instructions8.status = STARTED
    thisExp.addData('instructions8.started', instructions8.tStart)
    instructions8.maxDuration = None
    # keep track of which components have finished
    instructions8Components = instructions8.components
    for thisComponent in instructions8.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions8" ---
    instructions8.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_text_3* updates
        
        # if instructions_text_3 is starting this frame...
        if instructions_text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_text_3.frameNStart = frameN  # exact frame index
            instructions_text_3.tStart = t  # local t and not account for scr refresh
            instructions_text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_text_3.started')
            # update status
            instructions_text_3.status = STARTED
            instructions_text_3.setAutoDraw(True)
        
        # if instructions_text_3 is active this frame...
        if instructions_text_3.status == STARTED:
            # update params
            pass
        
        # *Continue_txt_3* updates
        
        # if Continue_txt_3 is starting this frame...
        if Continue_txt_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Continue_txt_3.frameNStart = frameN  # exact frame index
            Continue_txt_3.tStart = t  # local t and not account for scr refresh
            Continue_txt_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Continue_txt_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Continue_txt_3.started')
            # update status
            Continue_txt_3.status = STARTED
            Continue_txt_3.setAutoDraw(True)
        
        # if Continue_txt_3 is active this frame...
        if Continue_txt_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst_7* updates
        waitOnFlip = False
        
        # if key_resp_inst_7 is starting this frame...
        if key_resp_inst_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst_7.frameNStart = frameN  # exact frame index
            key_resp_inst_7.tStart = t  # local t and not account for scr refresh
            key_resp_inst_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_inst_7.started')
            # update status
            key_resp_inst_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst_7.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst_7.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_7_allKeys.extend(theseKeys)
            if len(_key_resp_inst_7_allKeys):
                key_resp_inst_7.keys = _key_resp_inst_7_allKeys[-1].name  # just the last key pressed
                key_resp_inst_7.rt = _key_resp_inst_7_allKeys[-1].rt
                key_resp_inst_7.duration = _key_resp_inst_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions8.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions8.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions8" ---
    for thisComponent in instructions8.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions8
    instructions8.tStop = globalClock.getTime(format='float')
    instructions8.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions8.stopped', instructions8.tStop)
    # check responses
    if key_resp_inst_7.keys in ['', [], None]:  # No response was made
        key_resp_inst_7.keys = None
    thisExp.addData('key_resp_inst_7.keys',key_resp_inst_7.keys)
    if key_resp_inst_7.keys != None:  # we had a response
        thisExp.addData('key_resp_inst_7.rt', key_resp_inst_7.rt)
        thisExp.addData('key_resp_inst_7.duration', key_resp_inst_7.duration)
    thisExp.nextEntry()
    # the Routine "instructions8" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "happy_overall" ---
    # create an object to store info about Routine happy_overall
    happy_overall = data.Routine(
        name='happy_overall',
        components=[question_3, happiness_rating_overall, low_end_text_3, high_end_text_3, key_resp_7, exit_text_4],
    )
    happy_overall.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    happiness_rating_overall.reset()
    # create starting attributes for key_resp_7
    key_resp_7.keys = []
    key_resp_7.rt = []
    _key_resp_7_allKeys = []
    # store start times for happy_overall
    happy_overall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    happy_overall.tStart = globalClock.getTime(format='float')
    happy_overall.status = STARTED
    thisExp.addData('happy_overall.started', happy_overall.tStart)
    happy_overall.maxDuration = None
    # keep track of which components have finished
    happy_overallComponents = happy_overall.components
    for thisComponent in happy_overall.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "happy_overall" ---
    happy_overall.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *question_3* updates
        
        # if question_3 is starting this frame...
        if question_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_3.frameNStart = frameN  # exact frame index
            question_3.tStart = t  # local t and not account for scr refresh
            question_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'question_3.started')
            # update status
            question_3.status = STARTED
            question_3.setAutoDraw(True)
        
        # if question_3 is active this frame...
        if question_3.status == STARTED:
            # update params
            pass
        
        # *happiness_rating_overall* updates
        
        # if happiness_rating_overall is starting this frame...
        if happiness_rating_overall.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            happiness_rating_overall.frameNStart = frameN  # exact frame index
            happiness_rating_overall.tStart = t  # local t and not account for scr refresh
            happiness_rating_overall.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(happiness_rating_overall, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'happiness_rating_overall.started')
            # update status
            happiness_rating_overall.status = STARTED
            happiness_rating_overall.setAutoDraw(True)
        
        # if happiness_rating_overall is active this frame...
        if happiness_rating_overall.status == STARTED:
            # update params
            pass
        
        # *low_end_text_3* updates
        
        # if low_end_text_3 is starting this frame...
        if low_end_text_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            low_end_text_3.frameNStart = frameN  # exact frame index
            low_end_text_3.tStart = t  # local t and not account for scr refresh
            low_end_text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(low_end_text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'low_end_text_3.started')
            # update status
            low_end_text_3.status = STARTED
            low_end_text_3.setAutoDraw(True)
        
        # if low_end_text_3 is active this frame...
        if low_end_text_3.status == STARTED:
            # update params
            pass
        
        # *high_end_text_3* updates
        
        # if high_end_text_3 is starting this frame...
        if high_end_text_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            high_end_text_3.frameNStart = frameN  # exact frame index
            high_end_text_3.tStart = t  # local t and not account for scr refresh
            high_end_text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(high_end_text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'high_end_text_3.started')
            # update status
            high_end_text_3.status = STARTED
            high_end_text_3.setAutoDraw(True)
        
        # if high_end_text_3 is active this frame...
        if high_end_text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_7* updates
        waitOnFlip = False
        
        # if key_resp_7 is starting this frame...
        if key_resp_7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_7.frameNStart = frameN  # exact frame index
            key_resp_7.tStart = t  # local t and not account for scr refresh
            key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_7.started')
            # update status
            key_resp_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_7.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_7.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_7_allKeys.extend(theseKeys)
            if len(_key_resp_7_allKeys):
                key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                key_resp_7.duration = _key_resp_7_allKeys[-1].duration
        
        # *exit_text_4* updates
        
        # if exit_text_4 is starting this frame...
        if exit_text_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            exit_text_4.frameNStart = frameN  # exact frame index
            exit_text_4.tStart = t  # local t and not account for scr refresh
            exit_text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exit_text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'exit_text_4.started')
            # update status
            exit_text_4.status = STARTED
            exit_text_4.setAutoDraw(True)
        
        # if exit_text_4 is active this frame...
        if exit_text_4.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from code_4
        if happiness_rating_overall.getRating() != undefined  and events.getKeys() == "return" :
            continueRoutine = false
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            happy_overall.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in happy_overall.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "happy_overall" ---
    for thisComponent in happy_overall.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for happy_overall
    happy_overall.tStop = globalClock.getTime(format='float')
    happy_overall.tStopRefresh = tThisFlipGlobal
    thisExp.addData('happy_overall.stopped', happy_overall.tStop)
    thisExp.addData('happiness_rating_overall.response', happiness_rating_overall.getRating())
    thisExp.addData('happiness_rating_overall.rt', happiness_rating_overall.getRT())
    # check responses
    if key_resp_7.keys in ['', [], None]:  # No response was made
        key_resp_7.keys = None
    thisExp.addData('key_resp_7.keys',key_resp_7.keys)
    if key_resp_7.keys != None:  # we had a response
        thisExp.addData('key_resp_7.rt', key_resp_7.rt)
        thisExp.addData('key_resp_7.duration', key_resp_7.duration)
    thisExp.nextEntry()
    # the Routine "happy_overall" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions10" ---
    # create an object to store info about Routine instructions10
    instructions10 = data.Routine(
        name='instructions10',
        components=[instructions_text_6, Continue_txt_4, key_resp_inst_8],
    )
    instructions10.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_inst_8
    key_resp_inst_8.keys = []
    key_resp_inst_8.rt = []
    _key_resp_inst_8_allKeys = []
    # store start times for instructions10
    instructions10.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions10.tStart = globalClock.getTime(format='float')
    instructions10.status = STARTED
    thisExp.addData('instructions10.started', instructions10.tStart)
    instructions10.maxDuration = None
    # keep track of which components have finished
    instructions10Components = instructions10.components
    for thisComponent in instructions10.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions10" ---
    instructions10.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_text_6* updates
        
        # if instructions_text_6 is starting this frame...
        if instructions_text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_text_6.frameNStart = frameN  # exact frame index
            instructions_text_6.tStart = t  # local t and not account for scr refresh
            instructions_text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_text_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_text_6.started')
            # update status
            instructions_text_6.status = STARTED
            instructions_text_6.setAutoDraw(True)
        
        # if instructions_text_6 is active this frame...
        if instructions_text_6.status == STARTED:
            # update params
            pass
        
        # *Continue_txt_4* updates
        
        # if Continue_txt_4 is starting this frame...
        if Continue_txt_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Continue_txt_4.frameNStart = frameN  # exact frame index
            Continue_txt_4.tStart = t  # local t and not account for scr refresh
            Continue_txt_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Continue_txt_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Continue_txt_4.started')
            # update status
            Continue_txt_4.status = STARTED
            Continue_txt_4.setAutoDraw(True)
        
        # if Continue_txt_4 is active this frame...
        if Continue_txt_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst_8* updates
        waitOnFlip = False
        
        # if key_resp_inst_8 is starting this frame...
        if key_resp_inst_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst_8.frameNStart = frameN  # exact frame index
            key_resp_inst_8.tStart = t  # local t and not account for scr refresh
            key_resp_inst_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_inst_8.started')
            # update status
            key_resp_inst_8.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst_8.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst_8.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst_8.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_8_allKeys.extend(theseKeys)
            if len(_key_resp_inst_8_allKeys):
                key_resp_inst_8.keys = _key_resp_inst_8_allKeys[-1].name  # just the last key pressed
                key_resp_inst_8.rt = _key_resp_inst_8_allKeys[-1].rt
                key_resp_inst_8.duration = _key_resp_inst_8_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions10.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions10.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions10" ---
    for thisComponent in instructions10.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions10
    instructions10.tStop = globalClock.getTime(format='float')
    instructions10.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions10.stopped', instructions10.tStop)
    # check responses
    if key_resp_inst_8.keys in ['', [], None]:  # No response was made
        key_resp_inst_8.keys = None
    thisExp.addData('key_resp_inst_8.keys',key_resp_inst_8.keys)
    if key_resp_inst_8.keys != None:  # we had a response
        thisExp.addData('key_resp_inst_8.rt', key_resp_inst_8.rt)
        thisExp.addData('key_resp_inst_8.duration', key_resp_inst_8.duration)
    thisExp.nextEntry()
    # the Routine "instructions10" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "happy_base" ---
    # create an object to store info about Routine happy_base
    happy_base = data.Routine(
        name='happy_base',
        components=[happiness_rating_baseline, low_end_text_5, high_end_text_5, question_5, exit_text_3, key_resp_8],
    )
    happy_base.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    happiness_rating_baseline.reset()
    # create starting attributes for key_resp_8
    key_resp_8.keys = []
    key_resp_8.rt = []
    _key_resp_8_allKeys = []
    # store start times for happy_base
    happy_base.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    happy_base.tStart = globalClock.getTime(format='float')
    happy_base.status = STARTED
    thisExp.addData('happy_base.started', happy_base.tStart)
    happy_base.maxDuration = None
    # keep track of which components have finished
    happy_baseComponents = happy_base.components
    for thisComponent in happy_base.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "happy_base" ---
    happy_base.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *happiness_rating_baseline* updates
        
        # if happiness_rating_baseline is starting this frame...
        if happiness_rating_baseline.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
            # keep track of start time/frame for later
            happiness_rating_baseline.frameNStart = frameN  # exact frame index
            happiness_rating_baseline.tStart = t  # local t and not account for scr refresh
            happiness_rating_baseline.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(happiness_rating_baseline, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'happiness_rating_baseline.started')
            # update status
            happiness_rating_baseline.status = STARTED
            happiness_rating_baseline.setAutoDraw(True)
        
        # if happiness_rating_baseline is active this frame...
        if happiness_rating_baseline.status == STARTED:
            # update params
            pass
        
        # *low_end_text_5* updates
        
        # if low_end_text_5 is starting this frame...
        if low_end_text_5.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
            # keep track of start time/frame for later
            low_end_text_5.frameNStart = frameN  # exact frame index
            low_end_text_5.tStart = t  # local t and not account for scr refresh
            low_end_text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(low_end_text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'low_end_text_5.started')
            # update status
            low_end_text_5.status = STARTED
            low_end_text_5.setAutoDraw(True)
        
        # if low_end_text_5 is active this frame...
        if low_end_text_5.status == STARTED:
            # update params
            pass
        
        # *high_end_text_5* updates
        
        # if high_end_text_5 is starting this frame...
        if high_end_text_5.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
            # keep track of start time/frame for later
            high_end_text_5.frameNStart = frameN  # exact frame index
            high_end_text_5.tStart = t  # local t and not account for scr refresh
            high_end_text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(high_end_text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'high_end_text_5.started')
            # update status
            high_end_text_5.status = STARTED
            high_end_text_5.setAutoDraw(True)
        
        # if high_end_text_5 is active this frame...
        if high_end_text_5.status == STARTED:
            # update params
            pass
        
        # *question_5* updates
        
        # if question_5 is starting this frame...
        if question_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_5.frameNStart = frameN  # exact frame index
            question_5.tStart = t  # local t and not account for scr refresh
            question_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'question_5.started')
            # update status
            question_5.status = STARTED
            question_5.setAutoDraw(True)
        
        # if question_5 is active this frame...
        if question_5.status == STARTED:
            # update params
            pass
        
        # *exit_text_3* updates
        
        # if exit_text_3 is starting this frame...
        if exit_text_3.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
            # keep track of start time/frame for later
            exit_text_3.frameNStart = frameN  # exact frame index
            exit_text_3.tStart = t  # local t and not account for scr refresh
            exit_text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exit_text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'exit_text_3.started')
            # update status
            exit_text_3.status = STARTED
            exit_text_3.setAutoDraw(True)
        
        # if exit_text_3 is active this frame...
        if exit_text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_8* updates
        waitOnFlip = False
        
        # if key_resp_8 is starting this frame...
        if key_resp_8.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
            # keep track of start time/frame for later
            key_resp_8.frameNStart = frameN  # exact frame index
            key_resp_8.tStart = t  # local t and not account for scr refresh
            key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_8.started')
            # update status
            key_resp_8.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_8.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_8.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_8_allKeys.extend(theseKeys)
            if len(_key_resp_8_allKeys):
                key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                key_resp_8.duration = _key_resp_8_allKeys[-1].duration
        # Run 'Each Frame' code from code
        if happiness_rating_overall.getRating() != undefined  and events.getKeys() == "return" :
            continueRoutine = false
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            happy_base.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in happy_base.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "happy_base" ---
    for thisComponent in happy_base.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for happy_base
    happy_base.tStop = globalClock.getTime(format='float')
    happy_base.tStopRefresh = tThisFlipGlobal
    thisExp.addData('happy_base.stopped', happy_base.tStop)
    thisExp.addData('happiness_rating_baseline.response', happiness_rating_baseline.getRating())
    thisExp.addData('happiness_rating_baseline.rt', happiness_rating_baseline.getRT())
    # check responses
    if key_resp_8.keys in ['', [], None]:  # No response was made
        key_resp_8.keys = None
    thisExp.addData('key_resp_8.keys',key_resp_8.keys)
    if key_resp_8.keys != None:  # we had a response
        thisExp.addData('key_resp_8.rt', key_resp_8.rt)
        thisExp.addData('key_resp_8.duration', key_resp_8.duration)
    thisExp.nextEntry()
    # the Routine "happy_base" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions10_2" ---
    # create an object to store info about Routine instructions10_2
    instructions10_2 = data.Routine(
        name='instructions10_2',
        components=[instructions_text_2, Continue_txt_6, key_resp_inst_11],
    )
    instructions10_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_inst_11
    key_resp_inst_11.keys = []
    key_resp_inst_11.rt = []
    _key_resp_inst_11_allKeys = []
    # store start times for instructions10_2
    instructions10_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions10_2.tStart = globalClock.getTime(format='float')
    instructions10_2.status = STARTED
    thisExp.addData('instructions10_2.started', instructions10_2.tStart)
    instructions10_2.maxDuration = None
    # keep track of which components have finished
    instructions10_2Components = instructions10_2.components
    for thisComponent in instructions10_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions10_2" ---
    instructions10_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_text_2* updates
        
        # if instructions_text_2 is starting this frame...
        if instructions_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_text_2.frameNStart = frameN  # exact frame index
            instructions_text_2.tStart = t  # local t and not account for scr refresh
            instructions_text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_text_2.started')
            # update status
            instructions_text_2.status = STARTED
            instructions_text_2.setAutoDraw(True)
        
        # if instructions_text_2 is active this frame...
        if instructions_text_2.status == STARTED:
            # update params
            pass
        
        # *Continue_txt_6* updates
        
        # if Continue_txt_6 is starting this frame...
        if Continue_txt_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Continue_txt_6.frameNStart = frameN  # exact frame index
            Continue_txt_6.tStart = t  # local t and not account for scr refresh
            Continue_txt_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Continue_txt_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Continue_txt_6.started')
            # update status
            Continue_txt_6.status = STARTED
            Continue_txt_6.setAutoDraw(True)
        
        # if Continue_txt_6 is active this frame...
        if Continue_txt_6.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst_11* updates
        waitOnFlip = False
        
        # if key_resp_inst_11 is starting this frame...
        if key_resp_inst_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst_11.frameNStart = frameN  # exact frame index
            key_resp_inst_11.tStart = t  # local t and not account for scr refresh
            key_resp_inst_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_inst_11.started')
            # update status
            key_resp_inst_11.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst_11.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst_11.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst_11.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_11_allKeys.extend(theseKeys)
            if len(_key_resp_inst_11_allKeys):
                key_resp_inst_11.keys = _key_resp_inst_11_allKeys[-1].name  # just the last key pressed
                key_resp_inst_11.rt = _key_resp_inst_11_allKeys[-1].rt
                key_resp_inst_11.duration = _key_resp_inst_11_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions10_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions10_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions10_2" ---
    for thisComponent in instructions10_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions10_2
    instructions10_2.tStop = globalClock.getTime(format='float')
    instructions10_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions10_2.stopped', instructions10_2.tStop)
    # check responses
    if key_resp_inst_11.keys in ['', [], None]:  # No response was made
        key_resp_inst_11.keys = None
    thisExp.addData('key_resp_inst_11.keys',key_resp_inst_11.keys)
    if key_resp_inst_11.keys != None:  # we had a response
        thisExp.addData('key_resp_inst_11.rt', key_resp_inst_11.rt)
        thisExp.addData('key_resp_inst_11.duration', key_resp_inst_11.duration)
    thisExp.nextEntry()
    # the Routine "instructions10_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions11" ---
    # create an object to store info about Routine instructions11
    instructions11 = data.Routine(
        name='instructions11',
        components=[instructions_text_7, Continue_txt_5, key_resp_inst_9],
    )
    instructions11.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_inst_9
    key_resp_inst_9.keys = []
    key_resp_inst_9.rt = []
    _key_resp_inst_9_allKeys = []
    # store start times for instructions11
    instructions11.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions11.tStart = globalClock.getTime(format='float')
    instructions11.status = STARTED
    thisExp.addData('instructions11.started', instructions11.tStart)
    instructions11.maxDuration = None
    # keep track of which components have finished
    instructions11Components = instructions11.components
    for thisComponent in instructions11.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions11" ---
    instructions11.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_text_7* updates
        
        # if instructions_text_7 is starting this frame...
        if instructions_text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_text_7.frameNStart = frameN  # exact frame index
            instructions_text_7.tStart = t  # local t and not account for scr refresh
            instructions_text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_text_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_text_7.started')
            # update status
            instructions_text_7.status = STARTED
            instructions_text_7.setAutoDraw(True)
        
        # if instructions_text_7 is active this frame...
        if instructions_text_7.status == STARTED:
            # update params
            pass
        
        # *Continue_txt_5* updates
        
        # if Continue_txt_5 is starting this frame...
        if Continue_txt_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Continue_txt_5.frameNStart = frameN  # exact frame index
            Continue_txt_5.tStart = t  # local t and not account for scr refresh
            Continue_txt_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Continue_txt_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Continue_txt_5.started')
            # update status
            Continue_txt_5.status = STARTED
            Continue_txt_5.setAutoDraw(True)
        
        # if Continue_txt_5 is active this frame...
        if Continue_txt_5.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst_9* updates
        waitOnFlip = False
        
        # if key_resp_inst_9 is starting this frame...
        if key_resp_inst_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst_9.frameNStart = frameN  # exact frame index
            key_resp_inst_9.tStart = t  # local t and not account for scr refresh
            key_resp_inst_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_inst_9.started')
            # update status
            key_resp_inst_9.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst_9.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst_9.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst_9.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_9_allKeys.extend(theseKeys)
            if len(_key_resp_inst_9_allKeys):
                key_resp_inst_9.keys = _key_resp_inst_9_allKeys[-1].name  # just the last key pressed
                key_resp_inst_9.rt = _key_resp_inst_9_allKeys[-1].rt
                key_resp_inst_9.duration = _key_resp_inst_9_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions11.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions11.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions11" ---
    for thisComponent in instructions11.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions11
    instructions11.tStop = globalClock.getTime(format='float')
    instructions11.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions11.stopped', instructions11.tStop)
    # check responses
    if key_resp_inst_9.keys in ['', [], None]:  # No response was made
        key_resp_inst_9.keys = None
    thisExp.addData('key_resp_inst_9.keys',key_resp_inst_9.keys)
    if key_resp_inst_9.keys != None:  # we had a response
        thisExp.addData('key_resp_inst_9.rt', key_resp_inst_9.rt)
        thisExp.addData('key_resp_inst_9.duration', key_resp_inst_9.duration)
    thisExp.nextEntry()
    # the Routine "instructions11" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "first_delay" ---
    # create an object to store info about Routine first_delay
    first_delay = data.Routine(
        name='first_delay',
        components=[text_3],
    )
    first_delay.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for first_delay
    first_delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    first_delay.tStart = globalClock.getTime(format='float')
    first_delay.status = STARTED
    thisExp.addData('first_delay.started', first_delay.tStart)
    first_delay.maxDuration = time_first_delay
    # keep track of which components have finished
    first_delayComponents = first_delay.components
    for thisComponent in first_delay.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "first_delay" ---
    first_delay.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > first_delay.maxDuration-frameTolerance:
            first_delay.maxDurationReached = True
            continueRoutine = False
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            first_delay.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in first_delay.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "first_delay" ---
    for thisComponent in first_delay.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for first_delay
    first_delay.tStop = globalClock.getTime(format='float')
    first_delay.tStopRefresh = tThisFlipGlobal
    thisExp.addData('first_delay.stopped', first_delay.tStop)
    thisExp.nextEntry()
    # the Routine "first_delay" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_trials = data.TrialHandler2(
        name='practice_trials',
        nReps=npractice_trials*3, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(practice_trials)  # add the loop to the experiment
    thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
    if thisPractice_trial != None:
        for paramName in thisPractice_trial:
            globals()[paramName] = thisPractice_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_trial in practice_trials:
        currentLoop = practice_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        
        # --- Prepare to start Routine "p_options_show" ---
        # create an object to store info about Routine p_options_show
        p_options_show = data.Routine(
            name='p_options_show',
            components=[p_fixation_cross_4, p_box1, p_box2, p_box3, p_box4, p_box1_mag, p_box1_P, p_box2_mag, p_box2_P, p_box3_mag, p_box3_P, p_box4_mag, p_box4_P, p_choice],
        )
        p_options_show.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_9
        # Determine the type of trial
        
        if (options_prac[practice_trials.thisN,1]==1) and (options_prac[practice_trials.thisN,5]!=0): # Choice sure left
            sure_left = True
            forced_trial = False
        elif (options_prac[practice_trials.thisN,5]==1) and (options_prac[practice_trials.thisN,3]!=0):  # Choice sure right
            sure_left = False
            forced_trial = False
        elif (options_prac[practice_trials.thisN,0]==0) and (options_prac[practice_trials.thisN,1]==1): # forced gamble right
            trial_side_left = False
            forced_trial = True
            forced_type_sure = False
        elif (options_prac[practice_trials.thisN,4]==0) and (options_prac[practice_trials.thisN,5]==1): # forced gamble left
            trial_side_left = True
            forced_trial = True
            forced_type_sure = False
        elif (options_prac[practice_trials.thisN,1]==1): # forced sure left
            trial_side_left = True
            forced_trial = True
            forced_type_sure = True
        else: # forced sure right
            trial_side_left = False
            forced_trial = True
            forced_type_sure = True
        
        
        # Determine the coordonates and components of the boxes
        
        if not(forced_trial): # choice trial
            x1 = -width
            x2 = -width
            x3 = width
            x4 = width
            if sure_left: # sure option on left side
                y1 = 0
                y2 = 2 # Not showing
                y3 = height
                y4 = -height
                Mag1 = options_prac[practice_trials.thisN,0]
                P1 = ""
                Mag2 = 0
                P2 = 0
                Mag3 = options_prac[practice_trials.thisN,4]
                P3 = options_prac[practice_trials.thisN,5]
                Mag4 = options_prac[practice_trials.thisN,2]
                P4 = options_prac[practice_trials.thisN,3]
            else: # sure option on the right side
                y1 = height
                y2 = -height
                y3 = 0
                y4 = 2 # Not showing
                Mag1 = options_prac[practice_trials.thisN,2]
                P1 = options_prac[practice_trials.thisN,3]
                Mag2 = options_prac[practice_trials.thisN,0]
                P2 = options_prac[practice_trials.thisN,1]
                Mag3 = options_prac[practice_trials.thisN,4]
                P3 = options_prac[practice_trials.thisN,5]
                Mag4 = 0
                P4 = 0
        else: # forced option
            x1 = 0
            x2 = 0
            x3 = 0
            x4 = 0
            if forced_type_sure: # Forced sure option
                y1 = 0
                y2 = 2 # Not showing
                y3 = 2 # Not showing
                y4 = 2 # Not showing
                if trial_side_left: # option on the left
                    Mag1 = options_prac[practice_trials.thisN,0]
                    P1 = options_prac[practice_trials.thisN,1]
                else: # option on the right
                    Mag1 = options_prac[practice_trials.thisN,4]
                    P1 = options_prac[practice_trials.thisN,5]
                Mag2 = 0
                P2 = 0
                Mag3 = 0
                P3 = 0
                Mag4 = 0
                P4 = 0
            else: # Forced gamble option
                y1 = height
                y2 = -height
                y3 = 2 # Not showing
                y4 = 2 # Not showing
                if trial_side_left: # option on the left
                    Mag1 = options_prac[practice_trials.thisN,2]
                    P1 = options_prac[practice_trials.thisN,3]
                    Mag2 = options_prac[practice_trials.thisN,0]
                    P2 = options_prac[practice_trials.thisN,1]
                else: # option on the right
                    Mag1 = options_prac[practice_trials.thisN,4]
                    P1 = options_prac[practice_trials.thisN,5]
                    Mag2 = options_prac[practice_trials.thisN,2]
                    P2 = options_prac[practice_trials.thisN,3]
                Mag3 = 0
                P3 = 0
                Mag4 = 0
                P4 = 0
        p_box1.setPos((x1, y1))
        p_box2.setPos((x2, y2))
        p_box3.setPos((x3, y3))
        p_box4.setPos((x4, y4))
        p_box1_mag.setPos((x1, y1+0.03))
        p_box1_mag.setText(Mag1)
        p_box1_P.setPos((x1, y1-0.04))
        p_box1_P.setText(P1)
        p_box2_mag.setPos((x2, y2+0.03))
        p_box2_mag.setText(Mag2)
        p_box2_P.setPos((x2, y2-0.04))
        p_box2_P.setText(P2)
        p_box3_mag.setPos((x3, y3+0.03))
        p_box3_mag.setText(Mag3)
        p_box3_P.setPos((x3, y3-0.04))
        p_box3_P.setText(P3)
        p_box4_mag.setPos((x4, y4+0.03))
        p_box4_mag.setText(Mag4)
        p_box4_P.setPos((x4, y4-0.04))
        p_box4_P.setText(P4)
        # create starting attributes for p_choice
        p_choice.keys = []
        p_choice.rt = []
        _p_choice_allKeys = []
        # store start times for p_options_show
        p_options_show.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_options_show.tStart = globalClock.getTime(format='float')
        p_options_show.status = STARTED
        thisExp.addData('p_options_show.started', p_options_show.tStart)
        p_options_show.maxDuration = time_gamble_delay
        # keep track of which components have finished
        p_options_showComponents = p_options_show.components
        for thisComponent in p_options_show.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_options_show" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        p_options_show.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > p_options_show.maxDuration-frameTolerance:
                p_options_show.maxDurationReached = True
                continueRoutine = False
            
            # *p_fixation_cross_4* updates
            
            # if p_fixation_cross_4 is starting this frame...
            if p_fixation_cross_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_fixation_cross_4.frameNStart = frameN  # exact frame index
                p_fixation_cross_4.tStart = t  # local t and not account for scr refresh
                p_fixation_cross_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fixation_cross_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fixation_cross_4.started')
                # update status
                p_fixation_cross_4.status = STARTED
                p_fixation_cross_4.setAutoDraw(True)
            
            # if p_fixation_cross_4 is active this frame...
            if p_fixation_cross_4.status == STARTED:
                # update params
                pass
            
            # *p_box1* updates
            
            # if p_box1 is starting this frame...
            if p_box1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1.frameNStart = frameN  # exact frame index
                p_box1.tStart = t  # local t and not account for scr refresh
                p_box1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1.started')
                # update status
                p_box1.status = STARTED
                p_box1.setAutoDraw(True)
            
            # if p_box1 is active this frame...
            if p_box1.status == STARTED:
                # update params
                pass
            
            # *p_box2* updates
            
            # if p_box2 is starting this frame...
            if p_box2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2.frameNStart = frameN  # exact frame index
                p_box2.tStart = t  # local t and not account for scr refresh
                p_box2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2.started')
                # update status
                p_box2.status = STARTED
                p_box2.setAutoDraw(True)
            
            # if p_box2 is active this frame...
            if p_box2.status == STARTED:
                # update params
                pass
            
            # *p_box3* updates
            
            # if p_box3 is starting this frame...
            if p_box3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3.frameNStart = frameN  # exact frame index
                p_box3.tStart = t  # local t and not account for scr refresh
                p_box3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3.started')
                # update status
                p_box3.status = STARTED
                p_box3.setAutoDraw(True)
            
            # if p_box3 is active this frame...
            if p_box3.status == STARTED:
                # update params
                pass
            
            # *p_box4* updates
            
            # if p_box4 is starting this frame...
            if p_box4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4.frameNStart = frameN  # exact frame index
                p_box4.tStart = t  # local t and not account for scr refresh
                p_box4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4.started')
                # update status
                p_box4.status = STARTED
                p_box4.setAutoDraw(True)
            
            # if p_box4 is active this frame...
            if p_box4.status == STARTED:
                # update params
                pass
            
            # *p_box1_mag* updates
            
            # if p_box1_mag is starting this frame...
            if p_box1_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_mag.frameNStart = frameN  # exact frame index
                p_box1_mag.tStart = t  # local t and not account for scr refresh
                p_box1_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_mag.started')
                # update status
                p_box1_mag.status = STARTED
                p_box1_mag.setAutoDraw(True)
            
            # if p_box1_mag is active this frame...
            if p_box1_mag.status == STARTED:
                # update params
                pass
            
            # *p_box1_P* updates
            
            # if p_box1_P is starting this frame...
            if p_box1_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_P.frameNStart = frameN  # exact frame index
                p_box1_P.tStart = t  # local t and not account for scr refresh
                p_box1_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_P.started')
                # update status
                p_box1_P.status = STARTED
                p_box1_P.setAutoDraw(True)
            
            # if p_box1_P is active this frame...
            if p_box1_P.status == STARTED:
                # update params
                pass
            
            # *p_box2_mag* updates
            
            # if p_box2_mag is starting this frame...
            if p_box2_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_mag.frameNStart = frameN  # exact frame index
                p_box2_mag.tStart = t  # local t and not account for scr refresh
                p_box2_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_mag.started')
                # update status
                p_box2_mag.status = STARTED
                p_box2_mag.setAutoDraw(True)
            
            # if p_box2_mag is active this frame...
            if p_box2_mag.status == STARTED:
                # update params
                pass
            
            # *p_box2_P* updates
            
            # if p_box2_P is starting this frame...
            if p_box2_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_P.frameNStart = frameN  # exact frame index
                p_box2_P.tStart = t  # local t and not account for scr refresh
                p_box2_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_P.started')
                # update status
                p_box2_P.status = STARTED
                p_box2_P.setAutoDraw(True)
            
            # if p_box2_P is active this frame...
            if p_box2_P.status == STARTED:
                # update params
                pass
            
            # *p_box3_mag* updates
            
            # if p_box3_mag is starting this frame...
            if p_box3_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3_mag.frameNStart = frameN  # exact frame index
                p_box3_mag.tStart = t  # local t and not account for scr refresh
                p_box3_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3_mag.started')
                # update status
                p_box3_mag.status = STARTED
                p_box3_mag.setAutoDraw(True)
            
            # if p_box3_mag is active this frame...
            if p_box3_mag.status == STARTED:
                # update params
                pass
            
            # *p_box3_P* updates
            
            # if p_box3_P is starting this frame...
            if p_box3_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3_P.frameNStart = frameN  # exact frame index
                p_box3_P.tStart = t  # local t and not account for scr refresh
                p_box3_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3_P.started')
                # update status
                p_box3_P.status = STARTED
                p_box3_P.setAutoDraw(True)
            
            # if p_box3_P is active this frame...
            if p_box3_P.status == STARTED:
                # update params
                pass
            
            # *p_box4_mag* updates
            
            # if p_box4_mag is starting this frame...
            if p_box4_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4_mag.frameNStart = frameN  # exact frame index
                p_box4_mag.tStart = t  # local t and not account for scr refresh
                p_box4_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4_mag.started')
                # update status
                p_box4_mag.status = STARTED
                p_box4_mag.setAutoDraw(True)
            
            # if p_box4_mag is active this frame...
            if p_box4_mag.status == STARTED:
                # update params
                pass
            
            # *p_box4_P* updates
            
            # if p_box4_P is starting this frame...
            if p_box4_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4_P.frameNStart = frameN  # exact frame index
                p_box4_P.tStart = t  # local t and not account for scr refresh
                p_box4_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4_P.started')
                # update status
                p_box4_P.status = STARTED
                p_box4_P.setAutoDraw(True)
            
            # if p_box4_P is active this frame...
            if p_box4_P.status == STARTED:
                # update params
                pass
            
            # *p_choice* updates
            waitOnFlip = False
            
            # if p_choice is starting this frame...
            if p_choice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_choice.frameNStart = frameN  # exact frame index
                p_choice.tStart = t  # local t and not account for scr refresh
                p_choice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_choice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_choice.started')
                # update status
                p_choice.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(p_choice.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(p_choice.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if p_choice.status == STARTED and not waitOnFlip:
                theseKeys = p_choice.getKeys(keyList=['left', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _p_choice_allKeys.extend(theseKeys)
                if len(_p_choice_allKeys):
                    p_choice.keys = _p_choice_allKeys[-1].name  # just the last key pressed
                    p_choice.rt = _p_choice_allKeys[-1].rt
                    p_choice.duration = _p_choice_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_options_show.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_options_show.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_options_show" ---
        for thisComponent in p_options_show.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_options_show
        p_options_show.tStop = globalClock.getTime(format='float')
        p_options_show.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_options_show.stopped', p_options_show.tStop)
        # check responses
        if p_choice.keys in ['', [], None]:  # No response was made
            p_choice.keys = None
        practice_trials.addData('p_choice.keys',p_choice.keys)
        if p_choice.keys != None:  # we had a response
            practice_trials.addData('p_choice.rt', p_choice.rt)
            practice_trials.addData('p_choice.duration', p_choice.duration)
        # the Routine "p_options_show" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "p_chosen_option" ---
        # create an object to store info about Routine p_chosen_option
        p_chosen_option = data.Routine(
            name='p_chosen_option',
            components=[fixation_5, p_box1_2, p_box2_2, p_box3_2, p_box4_2, p_box1_mag_2, p_box1_P_2, p_box2_mag_2, p_box2_P_2],
        )
        p_chosen_option.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from p_chosen_option_code
        p_inputs = event.getKeys()
        p_userchoice = p_inputs[-1]
        
        if p_userchoice == 'left':
            y3 = 0
            y4 = 0
            choice = 0
        else:
            y1 = 0
            y2 = 0
            choice = 1
        
        thisExp.addData('choice_prac', choice)
        p_box1_2.setPos((x1, y1))
        p_box2_2.setPos((x2, y2))
        p_box3_2.setPos((x3, y3))
        p_box4_2.setPos((x4, y4))
        p_box1_mag_2.setPos((x1, y1+0.03))
        p_box1_mag_2.setText(Mag1)
        p_box1_P_2.setPos((x1, y1-0.04))
        p_box1_P_2.setText(P1)
        p_box2_mag_2.setPos((x2, y2+0.03))
        p_box2_mag_2.setText(Mag2)
        p_box2_P_2.setPos((x2, y2-0.04))
        p_box2_P_2.setText(P2)
        # store start times for p_chosen_option
        p_chosen_option.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_chosen_option.tStart = globalClock.getTime(format='float')
        p_chosen_option.status = STARTED
        thisExp.addData('p_chosen_option.started', p_chosen_option.tStart)
        p_chosen_option.maxDuration = time_chosen_gamble
        # keep track of which components have finished
        p_chosen_optionComponents = p_chosen_option.components
        for thisComponent in p_chosen_option.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_chosen_option" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        p_chosen_option.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > p_chosen_option.maxDuration-frameTolerance:
                p_chosen_option.maxDurationReached = True
                continueRoutine = False
            
            # *fixation_5* updates
            
            # if fixation_5 is starting this frame...
            if fixation_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_5.frameNStart = frameN  # exact frame index
                fixation_5.tStart = t  # local t and not account for scr refresh
                fixation_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_5.started')
                # update status
                fixation_5.status = STARTED
                fixation_5.setAutoDraw(True)
            
            # if fixation_5 is active this frame...
            if fixation_5.status == STARTED:
                # update params
                pass
            
            # *p_box1_2* updates
            
            # if p_box1_2 is starting this frame...
            if p_box1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_2.frameNStart = frameN  # exact frame index
                p_box1_2.tStart = t  # local t and not account for scr refresh
                p_box1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_2.started')
                # update status
                p_box1_2.status = STARTED
                p_box1_2.setAutoDraw(True)
            
            # if p_box1_2 is active this frame...
            if p_box1_2.status == STARTED:
                # update params
                pass
            
            # *p_box2_2* updates
            
            # if p_box2_2 is starting this frame...
            if p_box2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_2.frameNStart = frameN  # exact frame index
                p_box2_2.tStart = t  # local t and not account for scr refresh
                p_box2_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_2.started')
                # update status
                p_box2_2.status = STARTED
                p_box2_2.setAutoDraw(True)
            
            # if p_box2_2 is active this frame...
            if p_box2_2.status == STARTED:
                # update params
                pass
            
            # *p_box3_2* updates
            
            # if p_box3_2 is starting this frame...
            if p_box3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3_2.frameNStart = frameN  # exact frame index
                p_box3_2.tStart = t  # local t and not account for scr refresh
                p_box3_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3_2.started')
                # update status
                p_box3_2.status = STARTED
                p_box3_2.setAutoDraw(True)
            
            # if p_box3_2 is active this frame...
            if p_box3_2.status == STARTED:
                # update params
                pass
            
            # *p_box4_2* updates
            
            # if p_box4_2 is starting this frame...
            if p_box4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4_2.frameNStart = frameN  # exact frame index
                p_box4_2.tStart = t  # local t and not account for scr refresh
                p_box4_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4_2.started')
                # update status
                p_box4_2.status = STARTED
                p_box4_2.setAutoDraw(True)
            
            # if p_box4_2 is active this frame...
            if p_box4_2.status == STARTED:
                # update params
                pass
            
            # *p_box1_mag_2* updates
            
            # if p_box1_mag_2 is starting this frame...
            if p_box1_mag_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_mag_2.frameNStart = frameN  # exact frame index
                p_box1_mag_2.tStart = t  # local t and not account for scr refresh
                p_box1_mag_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_mag_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_mag_2.started')
                # update status
                p_box1_mag_2.status = STARTED
                p_box1_mag_2.setAutoDraw(True)
            
            # if p_box1_mag_2 is active this frame...
            if p_box1_mag_2.status == STARTED:
                # update params
                pass
            
            # *p_box1_P_2* updates
            
            # if p_box1_P_2 is starting this frame...
            if p_box1_P_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_P_2.frameNStart = frameN  # exact frame index
                p_box1_P_2.tStart = t  # local t and not account for scr refresh
                p_box1_P_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_P_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_P_2.started')
                # update status
                p_box1_P_2.status = STARTED
                p_box1_P_2.setAutoDraw(True)
            
            # if p_box1_P_2 is active this frame...
            if p_box1_P_2.status == STARTED:
                # update params
                pass
            
            # *p_box2_mag_2* updates
            
            # if p_box2_mag_2 is starting this frame...
            if p_box2_mag_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_mag_2.frameNStart = frameN  # exact frame index
                p_box2_mag_2.tStart = t  # local t and not account for scr refresh
                p_box2_mag_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_mag_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_mag_2.started')
                # update status
                p_box2_mag_2.status = STARTED
                p_box2_mag_2.setAutoDraw(True)
            
            # if p_box2_mag_2 is active this frame...
            if p_box2_mag_2.status == STARTED:
                # update params
                pass
            
            # *p_box2_P_2* updates
            
            # if p_box2_P_2 is starting this frame...
            if p_box2_P_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_P_2.frameNStart = frameN  # exact frame index
                p_box2_P_2.tStart = t  # local t and not account for scr refresh
                p_box2_P_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_P_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_P_2.started')
                # update status
                p_box2_P_2.status = STARTED
                p_box2_P_2.setAutoDraw(True)
            
            # if p_box2_P_2 is active this frame...
            if p_box2_P_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_chosen_option.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_chosen_option.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_chosen_option" ---
        for thisComponent in p_chosen_option.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_chosen_option
        p_chosen_option.tStop = globalClock.getTime(format='float')
        p_chosen_option.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_chosen_option.stopped', p_chosen_option.tStop)
        # the Routine "p_chosen_option" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "p_outcome" ---
        # create an object to store info about Routine p_outcome
        p_outcome = data.Routine(
            name='p_outcome',
            components=[fixation_6, p_money_prompt, p_current_money_txt, practice_txt, p_outcome_square, p_outcome_text, p_outcome_rewa_text, p_prog_bar, p_next_trial_txt, p_next_trial_input],
        )
        p_outcome.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from p_gamble_result_code
        p_progBar = p_progVal/npractice_trials
        p_progVal += 1
        
        event.clearEvents()
        
        thisExp.addData('current_money_prac', current_money)
        
        #if p_userchoice == 'left':
        #    if gambles[practice_trials.thisN][0] >= gamble_result[practice_trials.thisN]:
        #        current_money += gambles[practice_trials.thisN][1]
        #        outcome_txtW = 'Win!'
        #        W_pos = 0.03
        #        outcome_txtN = gambles_txt[practice_trials.thisN][1]
        #        N_pos = -0.03
        #        outcome_color = 'Green'
        #        outcome = 1
        #    else:
        #        outcome_txtW = 'Loss!'
        #        W_pos = 0
        #        outcome_txtN = ''
        #        N_pos = -1.5
        #        outcome_color = 'Red'
        #        outcome = 0
        #else:
        #    if gambles[practice_trials.thisN][2] >= gamble_result[practice_trials.thisN]:
        #        current_money += gambles[practice_trials.thisN][3]
        #        outcome_txtW = 'Win!'
        #        W_pos = 0.03
        #        outcome_txtN = gambles_txt[practice_trials.thisN][3]
        #        N_pos = -0.03
        #        outcome_color = 'Green'
        #        outcome = 1
        #    else:
        #        outcome_txtW = 'Loss!'
        #        W_pos = 0
        #        outcome_txtN = ''
        #        N_pos = -1.5
        #        outcome_color = 'Red'
        #        outcome = 0
        #p_money_txt = f"$ {current_money:.2f}"
        #
        #thisExp.addData("outcome_prac", outcome)
        p_current_money_txt.setText(p_money_txt)
        p_outcome_square.setFillColor(outcome_color)
        p_outcome_square.setPos((choice_pos, 0))
        p_outcome_square.setSize((2*gamble_size, 2*gamble_size))
        p_outcome_square.setLineColor(outcome_color)
        p_outcome_text.setPos((choice_pos, W_pos))
        p_outcome_text.setText(outcome_txtW)
        p_outcome_rewa_text.setPos((choice_pos, rewa_pos))
        p_outcome_rewa_text.setText(outcome_rewa + '\n' + outcome_rewaL_txt)
        p_prog_bar.setProgress(p_progBar)
        # create starting attributes for p_next_trial_input
        p_next_trial_input.keys = []
        p_next_trial_input.rt = []
        _p_next_trial_input_allKeys = []
        # store start times for p_outcome
        p_outcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_outcome.tStart = globalClock.getTime(format='float')
        p_outcome.status = STARTED
        thisExp.addData('p_outcome.started', p_outcome.tStart)
        p_outcome.maxDuration = None
        # keep track of which components have finished
        p_outcomeComponents = p_outcome.components
        for thisComponent in p_outcome.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_outcome" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        p_outcome.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_6* updates
            
            # if fixation_6 is starting this frame...
            if fixation_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_6.frameNStart = frameN  # exact frame index
                fixation_6.tStart = t  # local t and not account for scr refresh
                fixation_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_6.started')
                # update status
                fixation_6.status = STARTED
                fixation_6.setAutoDraw(True)
            
            # if fixation_6 is active this frame...
            if fixation_6.status == STARTED:
                # update params
                pass
            
            # *p_money_prompt* updates
            
            # if p_money_prompt is starting this frame...
            if p_money_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_money_prompt.frameNStart = frameN  # exact frame index
                p_money_prompt.tStart = t  # local t and not account for scr refresh
                p_money_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_money_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_money_prompt.started')
                # update status
                p_money_prompt.status = STARTED
                p_money_prompt.setAutoDraw(True)
            
            # if p_money_prompt is active this frame...
            if p_money_prompt.status == STARTED:
                # update params
                pass
            
            # *p_current_money_txt* updates
            
            # if p_current_money_txt is starting this frame...
            if p_current_money_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_current_money_txt.frameNStart = frameN  # exact frame index
                p_current_money_txt.tStart = t  # local t and not account for scr refresh
                p_current_money_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_current_money_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_current_money_txt.started')
                # update status
                p_current_money_txt.status = STARTED
                p_current_money_txt.setAutoDraw(True)
            
            # if p_current_money_txt is active this frame...
            if p_current_money_txt.status == STARTED:
                # update params
                pass
            
            # *practice_txt* updates
            
            # if practice_txt is starting this frame...
            if practice_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_txt.frameNStart = frameN  # exact frame index
                practice_txt.tStart = t  # local t and not account for scr refresh
                practice_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_txt.started')
                # update status
                practice_txt.status = STARTED
                practice_txt.setAutoDraw(True)
            
            # if practice_txt is active this frame...
            if practice_txt.status == STARTED:
                # update params
                pass
            
            # *p_outcome_square* updates
            
            # if p_outcome_square is starting this frame...
            if p_outcome_square.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_outcome_square.frameNStart = frameN  # exact frame index
                p_outcome_square.tStart = t  # local t and not account for scr refresh
                p_outcome_square.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_outcome_square, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_outcome_square.started')
                # update status
                p_outcome_square.status = STARTED
                p_outcome_square.setAutoDraw(True)
            
            # if p_outcome_square is active this frame...
            if p_outcome_square.status == STARTED:
                # update params
                pass
            
            # *p_outcome_text* updates
            
            # if p_outcome_text is starting this frame...
            if p_outcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_outcome_text.frameNStart = frameN  # exact frame index
                p_outcome_text.tStart = t  # local t and not account for scr refresh
                p_outcome_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_outcome_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_outcome_text.started')
                # update status
                p_outcome_text.status = STARTED
                p_outcome_text.setAutoDraw(True)
            
            # if p_outcome_text is active this frame...
            if p_outcome_text.status == STARTED:
                # update params
                pass
            
            # *p_outcome_rewa_text* updates
            
            # if p_outcome_rewa_text is starting this frame...
            if p_outcome_rewa_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_outcome_rewa_text.frameNStart = frameN  # exact frame index
                p_outcome_rewa_text.tStart = t  # local t and not account for scr refresh
                p_outcome_rewa_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_outcome_rewa_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_outcome_rewa_text.started')
                # update status
                p_outcome_rewa_text.status = STARTED
                p_outcome_rewa_text.setAutoDraw(True)
            
            # if p_outcome_rewa_text is active this frame...
            if p_outcome_rewa_text.status == STARTED:
                # update params
                pass
            
            # *p_prog_bar* updates
            
            # if p_prog_bar is starting this frame...
            if p_prog_bar.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                p_prog_bar.frameNStart = frameN  # exact frame index
                p_prog_bar.tStart = t  # local t and not account for scr refresh
                p_prog_bar.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_prog_bar, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_prog_bar.started')
                # update status
                p_prog_bar.status = STARTED
                p_prog_bar.setAutoDraw(True)
            
            # if p_prog_bar is active this frame...
            if p_prog_bar.status == STARTED:
                # update params
                pass
            
            # *p_next_trial_txt* updates
            
            # if p_next_trial_txt is starting this frame...
            if p_next_trial_txt.status == NOT_STARTED and tThisFlip >= time_gamble_result-frameTolerance:
                # keep track of start time/frame for later
                p_next_trial_txt.frameNStart = frameN  # exact frame index
                p_next_trial_txt.tStart = t  # local t and not account for scr refresh
                p_next_trial_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_next_trial_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_next_trial_txt.started')
                # update status
                p_next_trial_txt.status = STARTED
                p_next_trial_txt.setAutoDraw(True)
            
            # if p_next_trial_txt is active this frame...
            if p_next_trial_txt.status == STARTED:
                # update params
                pass
            
            # *p_next_trial_input* updates
            waitOnFlip = False
            
            # if p_next_trial_input is starting this frame...
            if p_next_trial_input.status == NOT_STARTED and tThisFlip >= time_gamble_result-frameTolerance:
                # keep track of start time/frame for later
                p_next_trial_input.frameNStart = frameN  # exact frame index
                p_next_trial_input.tStart = t  # local t and not account for scr refresh
                p_next_trial_input.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_next_trial_input, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_next_trial_input.started')
                # update status
                p_next_trial_input.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(p_next_trial_input.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(p_next_trial_input.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if p_next_trial_input.status == STARTED and not waitOnFlip:
                theseKeys = p_next_trial_input.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _p_next_trial_input_allKeys.extend(theseKeys)
                if len(_p_next_trial_input_allKeys):
                    p_next_trial_input.keys = _p_next_trial_input_allKeys[-1].name  # just the last key pressed
                    p_next_trial_input.rt = _p_next_trial_input_allKeys[-1].rt
                    p_next_trial_input.duration = _p_next_trial_input_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_outcome.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_outcome.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_outcome" ---
        for thisComponent in p_outcome.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_outcome
        p_outcome.tStop = globalClock.getTime(format='float')
        p_outcome.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_outcome.stopped', p_outcome.tStop)
        # check responses
        if p_next_trial_input.keys in ['', [], None]:  # No response was made
            p_next_trial_input.keys = None
        practice_trials.addData('p_next_trial_input.keys',p_next_trial_input.keys)
        if p_next_trial_input.keys != None:  # we had a response
            practice_trials.addData('p_next_trial_input.rt', p_next_trial_input.rt)
            practice_trials.addData('p_next_trial_input.duration', p_next_trial_input.duration)
        # the Routine "p_outcome" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "iti" ---
        # create an object to store info about Routine iti
        iti = data.Routine(
            name='iti',
            components=[text],
        )
        iti.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for iti
        iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti.tStart = globalClock.getTime(format='float')
        iti.status = STARTED
        thisExp.addData('iti.started', iti.tStart)
        iti.maxDuration = time_iti
        # keep track of which components have finished
        itiComponents = iti.components
        for thisComponent in iti.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        iti.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > iti.maxDuration-frameTolerance:
                iti.maxDurationReached = True
                continueRoutine = False
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti" ---
        for thisComponent in iti.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti
        iti.tStop = globalClock.getTime(format='float')
        iti.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti.stopped', iti.tStop)
        # the Routine "iti" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "p_happiness_trial" ---
        # create an object to store info about Routine p_happiness_trial
        p_happiness_trial = data.Routine(
            name='p_happiness_trial',
            components=[happiness_rating_prac, low_end_text_4, high_end_text_4, question_4, exit_text_2, key_resp_5],
        )
        p_happiness_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        happiness_rating_prac.reset()
        # create starting attributes for key_resp_5
        key_resp_5.keys = []
        key_resp_5.rt = []
        _key_resp_5_allKeys = []
        # store start times for p_happiness_trial
        p_happiness_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_happiness_trial.tStart = globalClock.getTime(format='float')
        p_happiness_trial.status = STARTED
        thisExp.addData('p_happiness_trial.started', p_happiness_trial.tStart)
        p_happiness_trial.maxDuration = None
        # skip Routine p_happiness_trial if its 'Skip if' condition is True
        p_happiness_trial.skipped = continueRoutine and not (not (happyTrial_prac[practice_trials.thisN]))
        continueRoutine = p_happiness_trial.skipped
        # keep track of which components have finished
        p_happiness_trialComponents = p_happiness_trial.components
        for thisComponent in p_happiness_trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_happiness_trial" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        p_happiness_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *happiness_rating_prac* updates
            
            # if happiness_rating_prac is starting this frame...
            if happiness_rating_prac.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                happiness_rating_prac.frameNStart = frameN  # exact frame index
                happiness_rating_prac.tStart = t  # local t and not account for scr refresh
                happiness_rating_prac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(happiness_rating_prac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'happiness_rating_prac.started')
                # update status
                happiness_rating_prac.status = STARTED
                happiness_rating_prac.setAutoDraw(True)
            
            # if happiness_rating_prac is active this frame...
            if happiness_rating_prac.status == STARTED:
                # update params
                pass
            
            # *low_end_text_4* updates
            
            # if low_end_text_4 is starting this frame...
            if low_end_text_4.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                low_end_text_4.frameNStart = frameN  # exact frame index
                low_end_text_4.tStart = t  # local t and not account for scr refresh
                low_end_text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(low_end_text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'low_end_text_4.started')
                # update status
                low_end_text_4.status = STARTED
                low_end_text_4.setAutoDraw(True)
            
            # if low_end_text_4 is active this frame...
            if low_end_text_4.status == STARTED:
                # update params
                pass
            
            # *high_end_text_4* updates
            
            # if high_end_text_4 is starting this frame...
            if high_end_text_4.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                high_end_text_4.frameNStart = frameN  # exact frame index
                high_end_text_4.tStart = t  # local t and not account for scr refresh
                high_end_text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(high_end_text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'high_end_text_4.started')
                # update status
                high_end_text_4.status = STARTED
                high_end_text_4.setAutoDraw(True)
            
            # if high_end_text_4 is active this frame...
            if high_end_text_4.status == STARTED:
                # update params
                pass
            
            # *question_4* updates
            
            # if question_4 is starting this frame...
            if question_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_4.frameNStart = frameN  # exact frame index
                question_4.tStart = t  # local t and not account for scr refresh
                question_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_4.started')
                # update status
                question_4.status = STARTED
                question_4.setAutoDraw(True)
            
            # if question_4 is active this frame...
            if question_4.status == STARTED:
                # update params
                pass
            
            # *exit_text_2* updates
            
            # if exit_text_2 is starting this frame...
            if exit_text_2.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                exit_text_2.frameNStart = frameN  # exact frame index
                exit_text_2.tStart = t  # local t and not account for scr refresh
                exit_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(exit_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exit_text_2.started')
                # update status
                exit_text_2.status = STARTED
                exit_text_2.setAutoDraw(True)
            
            # if exit_text_2 is active this frame...
            if exit_text_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_5* updates
            waitOnFlip = False
            
            # if key_resp_5 is starting this frame...
            if key_resp_5.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_5.frameNStart = frameN  # exact frame index
                key_resp_5.tStart = t  # local t and not account for scr refresh
                key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_5.started')
                # update status
                key_resp_5.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_5.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_5.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_5_allKeys.extend(theseKeys)
                if len(_key_resp_5_allKeys):
                    key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                    key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                    key_resp_5.duration = _key_resp_5_allKeys[-1].duration
            # Run 'Each Frame' code from code_5
            if happiness_rating_overall.getRating() != undefined  and events.getKeys() == "return" :
                continueRoutine = false
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_happiness_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_happiness_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_happiness_trial" ---
        for thisComponent in p_happiness_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_happiness_trial
        p_happiness_trial.tStop = globalClock.getTime(format='float')
        p_happiness_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_happiness_trial.stopped', p_happiness_trial.tStop)
        practice_trials.addData('happiness_rating_prac.response', happiness_rating_prac.getRating())
        practice_trials.addData('happiness_rating_prac.rt', happiness_rating_prac.getRT())
        # check responses
        if key_resp_5.keys in ['', [], None]:  # No response was made
            key_resp_5.keys = None
        practice_trials.addData('key_resp_5.keys',key_resp_5.keys)
        if key_resp_5.keys != None:  # we had a response
            practice_trials.addData('key_resp_5.rt', key_resp_5.rt)
            practice_trials.addData('key_resp_5.duration', key_resp_5.duration)
        # the Routine "p_happiness_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "p_h_iti" ---
        # create an object to store info about Routine p_h_iti
        p_h_iti = data.Routine(
            name='p_h_iti',
            components=[text_2],
        )
        p_h_iti.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for p_h_iti
        p_h_iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_h_iti.tStart = globalClock.getTime(format='float')
        p_h_iti.status = STARTED
        thisExp.addData('p_h_iti.started', p_h_iti.tStart)
        p_h_iti.maxDuration = time_iti
        # skip Routine p_h_iti if its 'Skip if' condition is True
        p_h_iti.skipped = continueRoutine and not (not (happyTrial_prac[practice_trials.thisN]))
        continueRoutine = p_h_iti.skipped
        # keep track of which components have finished
        p_h_itiComponents = p_h_iti.components
        for thisComponent in p_h_iti.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_h_iti" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        p_h_iti.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > p_h_iti.maxDuration-frameTolerance:
                p_h_iti.maxDurationReached = True
                continueRoutine = False
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_h_iti.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_h_iti.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_h_iti" ---
        for thisComponent in p_h_iti.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_h_iti
        p_h_iti.tStop = globalClock.getTime(format='float')
        p_h_iti.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_h_iti.stopped', p_h_iti.tStop)
        # the Routine "p_h_iti" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "repeat_practice" ---
        # create an object to store info about Routine repeat_practice
        repeat_practice = data.Routine(
            name='repeat_practice',
            components=[repeat_practice_txt, key_resp_9],
        )
        repeat_practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_9
        key_resp_9.keys = []
        key_resp_9.rt = []
        _key_resp_9_allKeys = []
        # store start times for repeat_practice
        repeat_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        repeat_practice.tStart = globalClock.getTime(format='float')
        repeat_practice.status = STARTED
        thisExp.addData('repeat_practice.started', repeat_practice.tStart)
        repeat_practice.maxDuration = None
        # skip Routine repeat_practice if its 'Skip if' condition is True
        repeat_practice.skipped = continueRoutine and not ((practice_trials.thisN+1)%(npractice_trials/3) != 0)
        continueRoutine = repeat_practice.skipped
        # keep track of which components have finished
        repeat_practiceComponents = repeat_practice.components
        for thisComponent in repeat_practice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "repeat_practice" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        repeat_practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *repeat_practice_txt* updates
            
            # if repeat_practice_txt is starting this frame...
            if repeat_practice_txt.status == NOT_STARTED and tThisFlip >= time_iti-frameTolerance:
                # keep track of start time/frame for later
                repeat_practice_txt.frameNStart = frameN  # exact frame index
                repeat_practice_txt.tStart = t  # local t and not account for scr refresh
                repeat_practice_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(repeat_practice_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'repeat_practice_txt.started')
                # update status
                repeat_practice_txt.status = STARTED
                repeat_practice_txt.setAutoDraw(True)
            
            # if repeat_practice_txt is active this frame...
            if repeat_practice_txt.status == STARTED:
                # update params
                pass
            
            # *key_resp_9* updates
            waitOnFlip = False
            
            # if key_resp_9 is starting this frame...
            if key_resp_9.status == NOT_STARTED and tThisFlip >= time_iti-frameTolerance:
                # keep track of start time/frame for later
                key_resp_9.frameNStart = frameN  # exact frame index
                key_resp_9.tStart = t  # local t and not account for scr refresh
                key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_9.started')
                # update status
                key_resp_9.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_9.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_9.getKeys(keyList=['c', 'r'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_9_allKeys.extend(theseKeys)
                if len(_key_resp_9_allKeys):
                    key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                    key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                    key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                repeat_practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in repeat_practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "repeat_practice" ---
        for thisComponent in repeat_practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for repeat_practice
        repeat_practice.tStop = globalClock.getTime(format='float')
        repeat_practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('repeat_practice.stopped', repeat_practice.tStop)
        # check responses
        if key_resp_9.keys in ['', [], None]:  # No response was made
            key_resp_9.keys = None
        practice_trials.addData('key_resp_9.keys',key_resp_9.keys)
        if key_resp_9.keys != None:  # we had a response
            practice_trials.addData('key_resp_9.rt', key_resp_9.rt)
            practice_trials.addData('key_resp_9.duration', key_resp_9.duration)
        # Run 'End Routine' code from saveChoice_15
        # For an unknown reason, psychopy routines do not skip the custom code part of a shipped rountine
        if (practice_trials.thisN +1) % (npractice_trials/3) == 0:
            keys = event.getKeys()
            Input = keys[-1]
        
            if Input == 'c' or (practice_trials.thisN +1) == npractice_trials:
                practice_trials.finished = True
        
        #elif userChoice == 'r':
        #    practice_trials.finished = False
        #print(event.getKeys)
        #print(userChoice)
        #print(userChoice == 'l')
        # the Routine "repeat_practice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "repeat_iti" ---
        # create an object to store info about Routine repeat_iti
        repeat_iti = data.Routine(
            name='repeat_iti',
            components=[text_5],
        )
        repeat_iti.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for repeat_iti
        repeat_iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        repeat_iti.tStart = globalClock.getTime(format='float')
        repeat_iti.status = STARTED
        thisExp.addData('repeat_iti.started', repeat_iti.tStart)
        repeat_iti.maxDuration = time_iti
        # skip Routine repeat_iti if its 'Skip if' condition is True
        repeat_iti.skipped = continueRoutine and not ((practice_trials.thisN+1)%(npractice_trials/3) != 0)
        continueRoutine = repeat_iti.skipped
        # keep track of which components have finished
        repeat_itiComponents = repeat_iti.components
        for thisComponent in repeat_iti.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "repeat_iti" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        repeat_iti.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > repeat_iti.maxDuration-frameTolerance:
                repeat_iti.maxDurationReached = True
                continueRoutine = False
            
            # *text_5* updates
            
            # if text_5 is starting this frame...
            if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_5.frameNStart = frameN  # exact frame index
                text_5.tStart = t  # local t and not account for scr refresh
                text_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.started')
                # update status
                text_5.status = STARTED
                text_5.setAutoDraw(True)
            
            # if text_5 is active this frame...
            if text_5.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                repeat_iti.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in repeat_iti.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "repeat_iti" ---
        for thisComponent in repeat_iti.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for repeat_iti
        repeat_iti.tStop = globalClock.getTime(format='float')
        repeat_iti.tStopRefresh = tThisFlipGlobal
        thisExp.addData('repeat_iti.stopped', repeat_iti.tStop)
        # the Routine "repeat_iti" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed npractice_trials*3 repeats of 'practice_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instructions12" ---
    # create an object to store info about Routine instructions12
    instructions12 = data.Routine(
        name='instructions12',
        components=[instructions_text_8, key_resp_inst_10],
    )
    instructions12.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_inst_10
    key_resp_inst_10.keys = []
    key_resp_inst_10.rt = []
    _key_resp_inst_10_allKeys = []
    # store start times for instructions12
    instructions12.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions12.tStart = globalClock.getTime(format='float')
    instructions12.status = STARTED
    thisExp.addData('instructions12.started', instructions12.tStart)
    instructions12.maxDuration = None
    # keep track of which components have finished
    instructions12Components = instructions12.components
    for thisComponent in instructions12.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions12" ---
    instructions12.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_text_8* updates
        
        # if instructions_text_8 is starting this frame...
        if instructions_text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_text_8.frameNStart = frameN  # exact frame index
            instructions_text_8.tStart = t  # local t and not account for scr refresh
            instructions_text_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_text_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_text_8.started')
            # update status
            instructions_text_8.status = STARTED
            instructions_text_8.setAutoDraw(True)
        
        # if instructions_text_8 is active this frame...
        if instructions_text_8.status == STARTED:
            # update params
            pass
        
        # *key_resp_inst_10* updates
        waitOnFlip = False
        
        # if key_resp_inst_10 is starting this frame...
        if key_resp_inst_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_inst_10.frameNStart = frameN  # exact frame index
            key_resp_inst_10.tStart = t  # local t and not account for scr refresh
            key_resp_inst_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_inst_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_inst_10.started')
            # update status
            key_resp_inst_10.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_inst_10.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_inst_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_inst_10.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_inst_10.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_inst_10_allKeys.extend(theseKeys)
            if len(_key_resp_inst_10_allKeys):
                key_resp_inst_10.keys = _key_resp_inst_10_allKeys[-1].name  # just the last key pressed
                key_resp_inst_10.rt = _key_resp_inst_10_allKeys[-1].rt
                key_resp_inst_10.duration = _key_resp_inst_10_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions12.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions12.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions12" ---
    for thisComponent in instructions12.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions12
    instructions12.tStop = globalClock.getTime(format='float')
    instructions12.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions12.stopped', instructions12.tStop)
    # check responses
    if key_resp_inst_10.keys in ['', [], None]:  # No response was made
        key_resp_inst_10.keys = None
    thisExp.addData('key_resp_inst_10.keys',key_resp_inst_10.keys)
    if key_resp_inst_10.keys != None:  # we had a response
        thisExp.addData('key_resp_inst_10.rt', key_resp_inst_10.rt)
        thisExp.addData('key_resp_inst_10.duration', key_resp_inst_10.duration)
    thisExp.nextEntry()
    # the Routine "instructions12" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "first_delay" ---
    # create an object to store info about Routine first_delay
    first_delay = data.Routine(
        name='first_delay',
        components=[text_3],
    )
    first_delay.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for first_delay
    first_delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    first_delay.tStart = globalClock.getTime(format='float')
    first_delay.status = STARTED
    thisExp.addData('first_delay.started', first_delay.tStart)
    first_delay.maxDuration = time_first_delay
    # keep track of which components have finished
    first_delayComponents = first_delay.components
    for thisComponent in first_delay.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "first_delay" ---
    first_delay.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > first_delay.maxDuration-frameTolerance:
            first_delay.maxDurationReached = True
            continueRoutine = False
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            first_delay.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in first_delay.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "first_delay" ---
    for thisComponent in first_delay.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for first_delay
    first_delay.tStop = globalClock.getTime(format='float')
    first_delay.tStopRefresh = tThisFlipGlobal
    thisExp.addData('first_delay.stopped', first_delay.tStop)
    thisExp.nextEntry()
    # the Routine "first_delay" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=ntrials, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "gamble1_show" ---
        # create an object to store info about Routine gamble1_show
        gamble1_show = data.Routine(
            name='gamble1_show',
            components=[fixation_cross, gamble1_rew_text, gamble1_rew_text_low, gamble1_prob_text],
        )
        gamble1_show.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from gamble1_code
        thisExp.addData("dateTime", time.strftime("%Y-%m-%d_%H.%M.%S")) 
        
        gamble1_p = gambles[(trials.thisN+npractice_trials*3)][0]
        gamble1_r = gambles[(trials.thisN+npractice_trials*3)][1]
        
        gamble1_size = gamble1_r*pie_size
        
        Gamble1_green = visual.Pie(win, fillColor = 'Green', colorSpace = 'rgb',start = 0,end = 3.6*gamble1_p,pos=(-width,0), radius=gamble1_size)
        Gamble1_red = visual.Pie(win, fillColor = (-0.2,-0.2,-0.2), colorSpace = 'rgb',start = 3.6*gamble1_p,end = 360, pos=(-width,0), radius=gamble1_size)
        
        timer = core.CountdownTimer(start = time_gamble_show)
        
        thisExp.addData("gamble1_p", gamble1_p) 
        thisExp.addData("gamble1_r", gamble1_r)
        gamble1_rew_text.setText(gambles_txt[trials.thisN][1])
        gamble1_rew_text_low.setText(gambles_txt[trials.thisN][2])
        gamble1_prob_text.setText(gambles_txt[trials.thisN][0])
        # store start times for gamble1_show
        gamble1_show.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        gamble1_show.tStart = globalClock.getTime(format='float')
        gamble1_show.status = STARTED
        thisExp.addData('gamble1_show.started', gamble1_show.tStart)
        gamble1_show.maxDuration = time_gamble_show
        # keep track of which components have finished
        gamble1_showComponents = gamble1_show.components
        for thisComponent in gamble1_show.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "gamble1_show" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        gamble1_show.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > gamble1_show.maxDuration-frameTolerance:
                gamble1_show.maxDurationReached = True
                continueRoutine = False
            # Run 'Each Frame' code from gamble1_code
            if timer.getTime()>0: # delay before previous routine + delay after showing the gamble
                Gamble1_green.draw()
                Gamble1_red.draw()
            
            # *fixation_cross* updates
            
            # if fixation_cross is starting this frame...
            if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross.frameNStart = frameN  # exact frame index
                fixation_cross.tStart = t  # local t and not account for scr refresh
                fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross.started')
                # update status
                fixation_cross.status = STARTED
                fixation_cross.setAutoDraw(True)
            
            # if fixation_cross is active this frame...
            if fixation_cross.status == STARTED:
                # update params
                pass
            
            # *gamble1_rew_text* updates
            
            # if gamble1_rew_text is starting this frame...
            if gamble1_rew_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                gamble1_rew_text.frameNStart = frameN  # exact frame index
                gamble1_rew_text.tStart = t  # local t and not account for scr refresh
                gamble1_rew_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble1_rew_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble1_rew_text.started')
                # update status
                gamble1_rew_text.status = STARTED
                gamble1_rew_text.setAutoDraw(True)
            
            # if gamble1_rew_text is active this frame...
            if gamble1_rew_text.status == STARTED:
                # update params
                pass
            
            # *gamble1_rew_text_low* updates
            
            # if gamble1_rew_text_low is starting this frame...
            if gamble1_rew_text_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamble1_rew_text_low.frameNStart = frameN  # exact frame index
                gamble1_rew_text_low.tStart = t  # local t and not account for scr refresh
                gamble1_rew_text_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble1_rew_text_low, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble1_rew_text_low.started')
                # update status
                gamble1_rew_text_low.status = STARTED
                gamble1_rew_text_low.setAutoDraw(True)
            
            # if gamble1_rew_text_low is active this frame...
            if gamble1_rew_text_low.status == STARTED:
                # update params
                pass
            
            # *gamble1_prob_text* updates
            
            # if gamble1_prob_text is starting this frame...
            if gamble1_prob_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                gamble1_prob_text.frameNStart = frameN  # exact frame index
                gamble1_prob_text.tStart = t  # local t and not account for scr refresh
                gamble1_prob_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble1_prob_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble1_prob_text.started')
                # update status
                gamble1_prob_text.status = STARTED
                gamble1_prob_text.setAutoDraw(True)
            
            # if gamble1_prob_text is active this frame...
            if gamble1_prob_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                gamble1_show.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in gamble1_show.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "gamble1_show" ---
        for thisComponent in gamble1_show.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for gamble1_show
        gamble1_show.tStop = globalClock.getTime(format='float')
        gamble1_show.tStopRefresh = tThisFlipGlobal
        thisExp.addData('gamble1_show.stopped', gamble1_show.tStop)
        # the Routine "gamble1_show" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "p_options_show" ---
        # create an object to store info about Routine p_options_show
        p_options_show = data.Routine(
            name='p_options_show',
            components=[p_fixation_cross_4, p_box1, p_box2, p_box3, p_box4, p_box1_mag, p_box1_P, p_box2_mag, p_box2_P, p_box3_mag, p_box3_P, p_box4_mag, p_box4_P, p_choice],
        )
        p_options_show.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_9
        # Determine the type of trial
        
        if (options_prac[practice_trials.thisN,1]==1) and (options_prac[practice_trials.thisN,5]!=0): # Choice sure left
            sure_left = True
            forced_trial = False
        elif (options_prac[practice_trials.thisN,5]==1) and (options_prac[practice_trials.thisN,3]!=0):  # Choice sure right
            sure_left = False
            forced_trial = False
        elif (options_prac[practice_trials.thisN,0]==0) and (options_prac[practice_trials.thisN,1]==1): # forced gamble right
            trial_side_left = False
            forced_trial = True
            forced_type_sure = False
        elif (options_prac[practice_trials.thisN,4]==0) and (options_prac[practice_trials.thisN,5]==1): # forced gamble left
            trial_side_left = True
            forced_trial = True
            forced_type_sure = False
        elif (options_prac[practice_trials.thisN,1]==1): # forced sure left
            trial_side_left = True
            forced_trial = True
            forced_type_sure = True
        else: # forced sure right
            trial_side_left = False
            forced_trial = True
            forced_type_sure = True
        
        
        # Determine the coordonates and components of the boxes
        
        if not(forced_trial): # choice trial
            x1 = -width
            x2 = -width
            x3 = width
            x4 = width
            if sure_left: # sure option on left side
                y1 = 0
                y2 = 2 # Not showing
                y3 = height
                y4 = -height
                Mag1 = options_prac[practice_trials.thisN,0]
                P1 = ""
                Mag2 = 0
                P2 = 0
                Mag3 = options_prac[practice_trials.thisN,4]
                P3 = options_prac[practice_trials.thisN,5]
                Mag4 = options_prac[practice_trials.thisN,2]
                P4 = options_prac[practice_trials.thisN,3]
            else: # sure option on the right side
                y1 = height
                y2 = -height
                y3 = 0
                y4 = 2 # Not showing
                Mag1 = options_prac[practice_trials.thisN,2]
                P1 = options_prac[practice_trials.thisN,3]
                Mag2 = options_prac[practice_trials.thisN,0]
                P2 = options_prac[practice_trials.thisN,1]
                Mag3 = options_prac[practice_trials.thisN,4]
                P3 = options_prac[practice_trials.thisN,5]
                Mag4 = 0
                P4 = 0
        else: # forced option
            x1 = 0
            x2 = 0
            x3 = 0
            x4 = 0
            if forced_type_sure: # Forced sure option
                y1 = 0
                y2 = 2 # Not showing
                y3 = 2 # Not showing
                y4 = 2 # Not showing
                if trial_side_left: # option on the left
                    Mag1 = options_prac[practice_trials.thisN,0]
                    P1 = options_prac[practice_trials.thisN,1]
                else: # option on the right
                    Mag1 = options_prac[practice_trials.thisN,4]
                    P1 = options_prac[practice_trials.thisN,5]
                Mag2 = 0
                P2 = 0
                Mag3 = 0
                P3 = 0
                Mag4 = 0
                P4 = 0
            else: # Forced gamble option
                y1 = height
                y2 = -height
                y3 = 2 # Not showing
                y4 = 2 # Not showing
                if trial_side_left: # option on the left
                    Mag1 = options_prac[practice_trials.thisN,2]
                    P1 = options_prac[practice_trials.thisN,3]
                    Mag2 = options_prac[practice_trials.thisN,0]
                    P2 = options_prac[practice_trials.thisN,1]
                else: # option on the right
                    Mag1 = options_prac[practice_trials.thisN,4]
                    P1 = options_prac[practice_trials.thisN,5]
                    Mag2 = options_prac[practice_trials.thisN,2]
                    P2 = options_prac[practice_trials.thisN,3]
                Mag3 = 0
                P3 = 0
                Mag4 = 0
                P4 = 0
        p_box1.setPos((x1, y1))
        p_box2.setPos((x2, y2))
        p_box3.setPos((x3, y3))
        p_box4.setPos((x4, y4))
        p_box1_mag.setPos((x1, y1+0.03))
        p_box1_mag.setText(Mag1)
        p_box1_P.setPos((x1, y1-0.04))
        p_box1_P.setText(P1)
        p_box2_mag.setPos((x2, y2+0.03))
        p_box2_mag.setText(Mag2)
        p_box2_P.setPos((x2, y2-0.04))
        p_box2_P.setText(P2)
        p_box3_mag.setPos((x3, y3+0.03))
        p_box3_mag.setText(Mag3)
        p_box3_P.setPos((x3, y3-0.04))
        p_box3_P.setText(P3)
        p_box4_mag.setPos((x4, y4+0.03))
        p_box4_mag.setText(Mag4)
        p_box4_P.setPos((x4, y4-0.04))
        p_box4_P.setText(P4)
        # create starting attributes for p_choice
        p_choice.keys = []
        p_choice.rt = []
        _p_choice_allKeys = []
        # store start times for p_options_show
        p_options_show.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_options_show.tStart = globalClock.getTime(format='float')
        p_options_show.status = STARTED
        thisExp.addData('p_options_show.started', p_options_show.tStart)
        p_options_show.maxDuration = time_gamble_delay
        # keep track of which components have finished
        p_options_showComponents = p_options_show.components
        for thisComponent in p_options_show.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_options_show" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        p_options_show.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > p_options_show.maxDuration-frameTolerance:
                p_options_show.maxDurationReached = True
                continueRoutine = False
            
            # *p_fixation_cross_4* updates
            
            # if p_fixation_cross_4 is starting this frame...
            if p_fixation_cross_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_fixation_cross_4.frameNStart = frameN  # exact frame index
                p_fixation_cross_4.tStart = t  # local t and not account for scr refresh
                p_fixation_cross_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fixation_cross_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fixation_cross_4.started')
                # update status
                p_fixation_cross_4.status = STARTED
                p_fixation_cross_4.setAutoDraw(True)
            
            # if p_fixation_cross_4 is active this frame...
            if p_fixation_cross_4.status == STARTED:
                # update params
                pass
            
            # *p_box1* updates
            
            # if p_box1 is starting this frame...
            if p_box1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1.frameNStart = frameN  # exact frame index
                p_box1.tStart = t  # local t and not account for scr refresh
                p_box1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1.started')
                # update status
                p_box1.status = STARTED
                p_box1.setAutoDraw(True)
            
            # if p_box1 is active this frame...
            if p_box1.status == STARTED:
                # update params
                pass
            
            # *p_box2* updates
            
            # if p_box2 is starting this frame...
            if p_box2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2.frameNStart = frameN  # exact frame index
                p_box2.tStart = t  # local t and not account for scr refresh
                p_box2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2.started')
                # update status
                p_box2.status = STARTED
                p_box2.setAutoDraw(True)
            
            # if p_box2 is active this frame...
            if p_box2.status == STARTED:
                # update params
                pass
            
            # *p_box3* updates
            
            # if p_box3 is starting this frame...
            if p_box3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3.frameNStart = frameN  # exact frame index
                p_box3.tStart = t  # local t and not account for scr refresh
                p_box3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3.started')
                # update status
                p_box3.status = STARTED
                p_box3.setAutoDraw(True)
            
            # if p_box3 is active this frame...
            if p_box3.status == STARTED:
                # update params
                pass
            
            # *p_box4* updates
            
            # if p_box4 is starting this frame...
            if p_box4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4.frameNStart = frameN  # exact frame index
                p_box4.tStart = t  # local t and not account for scr refresh
                p_box4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4.started')
                # update status
                p_box4.status = STARTED
                p_box4.setAutoDraw(True)
            
            # if p_box4 is active this frame...
            if p_box4.status == STARTED:
                # update params
                pass
            
            # *p_box1_mag* updates
            
            # if p_box1_mag is starting this frame...
            if p_box1_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_mag.frameNStart = frameN  # exact frame index
                p_box1_mag.tStart = t  # local t and not account for scr refresh
                p_box1_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_mag.started')
                # update status
                p_box1_mag.status = STARTED
                p_box1_mag.setAutoDraw(True)
            
            # if p_box1_mag is active this frame...
            if p_box1_mag.status == STARTED:
                # update params
                pass
            
            # *p_box1_P* updates
            
            # if p_box1_P is starting this frame...
            if p_box1_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_P.frameNStart = frameN  # exact frame index
                p_box1_P.tStart = t  # local t and not account for scr refresh
                p_box1_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_P.started')
                # update status
                p_box1_P.status = STARTED
                p_box1_P.setAutoDraw(True)
            
            # if p_box1_P is active this frame...
            if p_box1_P.status == STARTED:
                # update params
                pass
            
            # *p_box2_mag* updates
            
            # if p_box2_mag is starting this frame...
            if p_box2_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_mag.frameNStart = frameN  # exact frame index
                p_box2_mag.tStart = t  # local t and not account for scr refresh
                p_box2_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_mag.started')
                # update status
                p_box2_mag.status = STARTED
                p_box2_mag.setAutoDraw(True)
            
            # if p_box2_mag is active this frame...
            if p_box2_mag.status == STARTED:
                # update params
                pass
            
            # *p_box2_P* updates
            
            # if p_box2_P is starting this frame...
            if p_box2_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_P.frameNStart = frameN  # exact frame index
                p_box2_P.tStart = t  # local t and not account for scr refresh
                p_box2_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_P.started')
                # update status
                p_box2_P.status = STARTED
                p_box2_P.setAutoDraw(True)
            
            # if p_box2_P is active this frame...
            if p_box2_P.status == STARTED:
                # update params
                pass
            
            # *p_box3_mag* updates
            
            # if p_box3_mag is starting this frame...
            if p_box3_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3_mag.frameNStart = frameN  # exact frame index
                p_box3_mag.tStart = t  # local t and not account for scr refresh
                p_box3_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3_mag.started')
                # update status
                p_box3_mag.status = STARTED
                p_box3_mag.setAutoDraw(True)
            
            # if p_box3_mag is active this frame...
            if p_box3_mag.status == STARTED:
                # update params
                pass
            
            # *p_box3_P* updates
            
            # if p_box3_P is starting this frame...
            if p_box3_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3_P.frameNStart = frameN  # exact frame index
                p_box3_P.tStart = t  # local t and not account for scr refresh
                p_box3_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3_P.started')
                # update status
                p_box3_P.status = STARTED
                p_box3_P.setAutoDraw(True)
            
            # if p_box3_P is active this frame...
            if p_box3_P.status == STARTED:
                # update params
                pass
            
            # *p_box4_mag* updates
            
            # if p_box4_mag is starting this frame...
            if p_box4_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4_mag.frameNStart = frameN  # exact frame index
                p_box4_mag.tStart = t  # local t and not account for scr refresh
                p_box4_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4_mag.started')
                # update status
                p_box4_mag.status = STARTED
                p_box4_mag.setAutoDraw(True)
            
            # if p_box4_mag is active this frame...
            if p_box4_mag.status == STARTED:
                # update params
                pass
            
            # *p_box4_P* updates
            
            # if p_box4_P is starting this frame...
            if p_box4_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4_P.frameNStart = frameN  # exact frame index
                p_box4_P.tStart = t  # local t and not account for scr refresh
                p_box4_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4_P.started')
                # update status
                p_box4_P.status = STARTED
                p_box4_P.setAutoDraw(True)
            
            # if p_box4_P is active this frame...
            if p_box4_P.status == STARTED:
                # update params
                pass
            
            # *p_choice* updates
            waitOnFlip = False
            
            # if p_choice is starting this frame...
            if p_choice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_choice.frameNStart = frameN  # exact frame index
                p_choice.tStart = t  # local t and not account for scr refresh
                p_choice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_choice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_choice.started')
                # update status
                p_choice.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(p_choice.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(p_choice.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if p_choice.status == STARTED and not waitOnFlip:
                theseKeys = p_choice.getKeys(keyList=['left', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _p_choice_allKeys.extend(theseKeys)
                if len(_p_choice_allKeys):
                    p_choice.keys = _p_choice_allKeys[-1].name  # just the last key pressed
                    p_choice.rt = _p_choice_allKeys[-1].rt
                    p_choice.duration = _p_choice_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_options_show.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_options_show.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_options_show" ---
        for thisComponent in p_options_show.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_options_show
        p_options_show.tStop = globalClock.getTime(format='float')
        p_options_show.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_options_show.stopped', p_options_show.tStop)
        # check responses
        if p_choice.keys in ['', [], None]:  # No response was made
            p_choice.keys = None
        trials.addData('p_choice.keys',p_choice.keys)
        if p_choice.keys != None:  # we had a response
            trials.addData('p_choice.rt', p_choice.rt)
            trials.addData('p_choice.duration', p_choice.duration)
        # the Routine "p_options_show" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "gamble2_show" ---
        # create an object to store info about Routine gamble2_show
        gamble2_show = data.Routine(
            name='gamble2_show',
            components=[fixation_cross_2, gamble2_rew_text, gamble2_rew_text_low, gamble2_prob_text],
        )
        gamble2_show.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from gamble2_code
        gamble2_p = gambles[(trials.thisN+npractice_trials*3)][2]
        gamble2_r = gambles[(trials.thisN+npractice_trials*3)][3]
        
        gamble2_size = gamble2_r*pie_size
        
        Gamble2_green = visual.Pie(win, fillColor = 'Green', colorSpace = 'rgb',start = 0,end = 3.6*gamble2_p,pos=(width,0), radius=gamble2_size)
        Gamble2_red = visual.Pie(win, fillColor = (-0.2,-0.2,-0.2), colorSpace = 'rgb',start = 3.6*gamble2_p,end = 360, pos=(width,0), radius=gamble2_size)
        
        timer = core.CountdownTimer(time_gamble_show+time_gamble_delay)
        
        thisExp.addData("gamble2_p", gamble2_p) 
        thisExp.addData("gamble2_r", gamble2_r)
        gamble2_rew_text.setText(gambles_txt[trials.thisN][4])
        gamble2_rew_text_low.setText(gambles_txt[trials.thisN][5])
        gamble2_prob_text.setText(gambles_txt[trials.thisN][3])
        # store start times for gamble2_show
        gamble2_show.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        gamble2_show.tStart = globalClock.getTime(format='float')
        gamble2_show.status = STARTED
        thisExp.addData('gamble2_show.started', gamble2_show.tStart)
        gamble2_show.maxDuration = time_gamble_show
        # keep track of which components have finished
        gamble2_showComponents = gamble2_show.components
        for thisComponent in gamble2_show.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "gamble2_show" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        gamble2_show.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > gamble2_show.maxDuration-frameTolerance:
                gamble2_show.maxDurationReached = True
                continueRoutine = False
            # Run 'Each Frame' code from gamble2_code
            if timer.getTime()>time_gamble_delay:
                Gamble2_green.draw()
                Gamble2_red.draw()
            
            # *fixation_cross_2* updates
            
            # if fixation_cross_2 is starting this frame...
            if fixation_cross_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross_2.frameNStart = frameN  # exact frame index
                fixation_cross_2.tStart = t  # local t and not account for scr refresh
                fixation_cross_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross_2.started')
                # update status
                fixation_cross_2.status = STARTED
                fixation_cross_2.setAutoDraw(True)
            
            # if fixation_cross_2 is active this frame...
            if fixation_cross_2.status == STARTED:
                # update params
                pass
            
            # *gamble2_rew_text* updates
            
            # if gamble2_rew_text is starting this frame...
            if gamble2_rew_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                gamble2_rew_text.frameNStart = frameN  # exact frame index
                gamble2_rew_text.tStart = t  # local t and not account for scr refresh
                gamble2_rew_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble2_rew_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble2_rew_text.started')
                # update status
                gamble2_rew_text.status = STARTED
                gamble2_rew_text.setAutoDraw(True)
            
            # if gamble2_rew_text is active this frame...
            if gamble2_rew_text.status == STARTED:
                # update params
                pass
            
            # *gamble2_rew_text_low* updates
            
            # if gamble2_rew_text_low is starting this frame...
            if gamble2_rew_text_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamble2_rew_text_low.frameNStart = frameN  # exact frame index
                gamble2_rew_text_low.tStart = t  # local t and not account for scr refresh
                gamble2_rew_text_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble2_rew_text_low, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble2_rew_text_low.started')
                # update status
                gamble2_rew_text_low.status = STARTED
                gamble2_rew_text_low.setAutoDraw(True)
            
            # if gamble2_rew_text_low is active this frame...
            if gamble2_rew_text_low.status == STARTED:
                # update params
                pass
            
            # *gamble2_prob_text* updates
            
            # if gamble2_prob_text is starting this frame...
            if gamble2_prob_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                gamble2_prob_text.frameNStart = frameN  # exact frame index
                gamble2_prob_text.tStart = t  # local t and not account for scr refresh
                gamble2_prob_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble2_prob_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble2_prob_text.started')
                # update status
                gamble2_prob_text.status = STARTED
                gamble2_prob_text.setAutoDraw(True)
            
            # if gamble2_prob_text is active this frame...
            if gamble2_prob_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                gamble2_show.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in gamble2_show.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "gamble2_show" ---
        for thisComponent in gamble2_show.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for gamble2_show
        gamble2_show.tStop = globalClock.getTime(format='float')
        gamble2_show.tStopRefresh = tThisFlipGlobal
        thisExp.addData('gamble2_show.stopped', gamble2_show.tStop)
        # the Routine "gamble2_show" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "p_options_show" ---
        # create an object to store info about Routine p_options_show
        p_options_show = data.Routine(
            name='p_options_show',
            components=[p_fixation_cross_4, p_box1, p_box2, p_box3, p_box4, p_box1_mag, p_box1_P, p_box2_mag, p_box2_P, p_box3_mag, p_box3_P, p_box4_mag, p_box4_P, p_choice],
        )
        p_options_show.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_9
        # Determine the type of trial
        
        if (options_prac[practice_trials.thisN,1]==1) and (options_prac[practice_trials.thisN,5]!=0): # Choice sure left
            sure_left = True
            forced_trial = False
        elif (options_prac[practice_trials.thisN,5]==1) and (options_prac[practice_trials.thisN,3]!=0):  # Choice sure right
            sure_left = False
            forced_trial = False
        elif (options_prac[practice_trials.thisN,0]==0) and (options_prac[practice_trials.thisN,1]==1): # forced gamble right
            trial_side_left = False
            forced_trial = True
            forced_type_sure = False
        elif (options_prac[practice_trials.thisN,4]==0) and (options_prac[practice_trials.thisN,5]==1): # forced gamble left
            trial_side_left = True
            forced_trial = True
            forced_type_sure = False
        elif (options_prac[practice_trials.thisN,1]==1): # forced sure left
            trial_side_left = True
            forced_trial = True
            forced_type_sure = True
        else: # forced sure right
            trial_side_left = False
            forced_trial = True
            forced_type_sure = True
        
        
        # Determine the coordonates and components of the boxes
        
        if not(forced_trial): # choice trial
            x1 = -width
            x2 = -width
            x3 = width
            x4 = width
            if sure_left: # sure option on left side
                y1 = 0
                y2 = 2 # Not showing
                y3 = height
                y4 = -height
                Mag1 = options_prac[practice_trials.thisN,0]
                P1 = ""
                Mag2 = 0
                P2 = 0
                Mag3 = options_prac[practice_trials.thisN,4]
                P3 = options_prac[practice_trials.thisN,5]
                Mag4 = options_prac[practice_trials.thisN,2]
                P4 = options_prac[practice_trials.thisN,3]
            else: # sure option on the right side
                y1 = height
                y2 = -height
                y3 = 0
                y4 = 2 # Not showing
                Mag1 = options_prac[practice_trials.thisN,2]
                P1 = options_prac[practice_trials.thisN,3]
                Mag2 = options_prac[practice_trials.thisN,0]
                P2 = options_prac[practice_trials.thisN,1]
                Mag3 = options_prac[practice_trials.thisN,4]
                P3 = options_prac[practice_trials.thisN,5]
                Mag4 = 0
                P4 = 0
        else: # forced option
            x1 = 0
            x2 = 0
            x3 = 0
            x4 = 0
            if forced_type_sure: # Forced sure option
                y1 = 0
                y2 = 2 # Not showing
                y3 = 2 # Not showing
                y4 = 2 # Not showing
                if trial_side_left: # option on the left
                    Mag1 = options_prac[practice_trials.thisN,0]
                    P1 = options_prac[practice_trials.thisN,1]
                else: # option on the right
                    Mag1 = options_prac[practice_trials.thisN,4]
                    P1 = options_prac[practice_trials.thisN,5]
                Mag2 = 0
                P2 = 0
                Mag3 = 0
                P3 = 0
                Mag4 = 0
                P4 = 0
            else: # Forced gamble option
                y1 = height
                y2 = -height
                y3 = 2 # Not showing
                y4 = 2 # Not showing
                if trial_side_left: # option on the left
                    Mag1 = options_prac[practice_trials.thisN,2]
                    P1 = options_prac[practice_trials.thisN,3]
                    Mag2 = options_prac[practice_trials.thisN,0]
                    P2 = options_prac[practice_trials.thisN,1]
                else: # option on the right
                    Mag1 = options_prac[practice_trials.thisN,4]
                    P1 = options_prac[practice_trials.thisN,5]
                    Mag2 = options_prac[practice_trials.thisN,2]
                    P2 = options_prac[practice_trials.thisN,3]
                Mag3 = 0
                P3 = 0
                Mag4 = 0
                P4 = 0
        p_box1.setPos((x1, y1))
        p_box2.setPos((x2, y2))
        p_box3.setPos((x3, y3))
        p_box4.setPos((x4, y4))
        p_box1_mag.setPos((x1, y1+0.03))
        p_box1_mag.setText(Mag1)
        p_box1_P.setPos((x1, y1-0.04))
        p_box1_P.setText(P1)
        p_box2_mag.setPos((x2, y2+0.03))
        p_box2_mag.setText(Mag2)
        p_box2_P.setPos((x2, y2-0.04))
        p_box2_P.setText(P2)
        p_box3_mag.setPos((x3, y3+0.03))
        p_box3_mag.setText(Mag3)
        p_box3_P.setPos((x3, y3-0.04))
        p_box3_P.setText(P3)
        p_box4_mag.setPos((x4, y4+0.03))
        p_box4_mag.setText(Mag4)
        p_box4_P.setPos((x4, y4-0.04))
        p_box4_P.setText(P4)
        # create starting attributes for p_choice
        p_choice.keys = []
        p_choice.rt = []
        _p_choice_allKeys = []
        # store start times for p_options_show
        p_options_show.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        p_options_show.tStart = globalClock.getTime(format='float')
        p_options_show.status = STARTED
        thisExp.addData('p_options_show.started', p_options_show.tStart)
        p_options_show.maxDuration = time_gamble_delay
        # keep track of which components have finished
        p_options_showComponents = p_options_show.components
        for thisComponent in p_options_show.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "p_options_show" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        p_options_show.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > p_options_show.maxDuration-frameTolerance:
                p_options_show.maxDurationReached = True
                continueRoutine = False
            
            # *p_fixation_cross_4* updates
            
            # if p_fixation_cross_4 is starting this frame...
            if p_fixation_cross_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_fixation_cross_4.frameNStart = frameN  # exact frame index
                p_fixation_cross_4.tStart = t  # local t and not account for scr refresh
                p_fixation_cross_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fixation_cross_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fixation_cross_4.started')
                # update status
                p_fixation_cross_4.status = STARTED
                p_fixation_cross_4.setAutoDraw(True)
            
            # if p_fixation_cross_4 is active this frame...
            if p_fixation_cross_4.status == STARTED:
                # update params
                pass
            
            # *p_box1* updates
            
            # if p_box1 is starting this frame...
            if p_box1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1.frameNStart = frameN  # exact frame index
                p_box1.tStart = t  # local t and not account for scr refresh
                p_box1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1.started')
                # update status
                p_box1.status = STARTED
                p_box1.setAutoDraw(True)
            
            # if p_box1 is active this frame...
            if p_box1.status == STARTED:
                # update params
                pass
            
            # *p_box2* updates
            
            # if p_box2 is starting this frame...
            if p_box2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2.frameNStart = frameN  # exact frame index
                p_box2.tStart = t  # local t and not account for scr refresh
                p_box2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2.started')
                # update status
                p_box2.status = STARTED
                p_box2.setAutoDraw(True)
            
            # if p_box2 is active this frame...
            if p_box2.status == STARTED:
                # update params
                pass
            
            # *p_box3* updates
            
            # if p_box3 is starting this frame...
            if p_box3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3.frameNStart = frameN  # exact frame index
                p_box3.tStart = t  # local t and not account for scr refresh
                p_box3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3.started')
                # update status
                p_box3.status = STARTED
                p_box3.setAutoDraw(True)
            
            # if p_box3 is active this frame...
            if p_box3.status == STARTED:
                # update params
                pass
            
            # *p_box4* updates
            
            # if p_box4 is starting this frame...
            if p_box4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4.frameNStart = frameN  # exact frame index
                p_box4.tStart = t  # local t and not account for scr refresh
                p_box4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4.started')
                # update status
                p_box4.status = STARTED
                p_box4.setAutoDraw(True)
            
            # if p_box4 is active this frame...
            if p_box4.status == STARTED:
                # update params
                pass
            
            # *p_box1_mag* updates
            
            # if p_box1_mag is starting this frame...
            if p_box1_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_mag.frameNStart = frameN  # exact frame index
                p_box1_mag.tStart = t  # local t and not account for scr refresh
                p_box1_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_mag.started')
                # update status
                p_box1_mag.status = STARTED
                p_box1_mag.setAutoDraw(True)
            
            # if p_box1_mag is active this frame...
            if p_box1_mag.status == STARTED:
                # update params
                pass
            
            # *p_box1_P* updates
            
            # if p_box1_P is starting this frame...
            if p_box1_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box1_P.frameNStart = frameN  # exact frame index
                p_box1_P.tStart = t  # local t and not account for scr refresh
                p_box1_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box1_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box1_P.started')
                # update status
                p_box1_P.status = STARTED
                p_box1_P.setAutoDraw(True)
            
            # if p_box1_P is active this frame...
            if p_box1_P.status == STARTED:
                # update params
                pass
            
            # *p_box2_mag* updates
            
            # if p_box2_mag is starting this frame...
            if p_box2_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_mag.frameNStart = frameN  # exact frame index
                p_box2_mag.tStart = t  # local t and not account for scr refresh
                p_box2_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_mag.started')
                # update status
                p_box2_mag.status = STARTED
                p_box2_mag.setAutoDraw(True)
            
            # if p_box2_mag is active this frame...
            if p_box2_mag.status == STARTED:
                # update params
                pass
            
            # *p_box2_P* updates
            
            # if p_box2_P is starting this frame...
            if p_box2_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box2_P.frameNStart = frameN  # exact frame index
                p_box2_P.tStart = t  # local t and not account for scr refresh
                p_box2_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box2_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box2_P.started')
                # update status
                p_box2_P.status = STARTED
                p_box2_P.setAutoDraw(True)
            
            # if p_box2_P is active this frame...
            if p_box2_P.status == STARTED:
                # update params
                pass
            
            # *p_box3_mag* updates
            
            # if p_box3_mag is starting this frame...
            if p_box3_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3_mag.frameNStart = frameN  # exact frame index
                p_box3_mag.tStart = t  # local t and not account for scr refresh
                p_box3_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3_mag.started')
                # update status
                p_box3_mag.status = STARTED
                p_box3_mag.setAutoDraw(True)
            
            # if p_box3_mag is active this frame...
            if p_box3_mag.status == STARTED:
                # update params
                pass
            
            # *p_box3_P* updates
            
            # if p_box3_P is starting this frame...
            if p_box3_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box3_P.frameNStart = frameN  # exact frame index
                p_box3_P.tStart = t  # local t and not account for scr refresh
                p_box3_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box3_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box3_P.started')
                # update status
                p_box3_P.status = STARTED
                p_box3_P.setAutoDraw(True)
            
            # if p_box3_P is active this frame...
            if p_box3_P.status == STARTED:
                # update params
                pass
            
            # *p_box4_mag* updates
            
            # if p_box4_mag is starting this frame...
            if p_box4_mag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4_mag.frameNStart = frameN  # exact frame index
                p_box4_mag.tStart = t  # local t and not account for scr refresh
                p_box4_mag.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4_mag, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4_mag.started')
                # update status
                p_box4_mag.status = STARTED
                p_box4_mag.setAutoDraw(True)
            
            # if p_box4_mag is active this frame...
            if p_box4_mag.status == STARTED:
                # update params
                pass
            
            # *p_box4_P* updates
            
            # if p_box4_P is starting this frame...
            if p_box4_P.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_box4_P.frameNStart = frameN  # exact frame index
                p_box4_P.tStart = t  # local t and not account for scr refresh
                p_box4_P.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_box4_P, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_box4_P.started')
                # update status
                p_box4_P.status = STARTED
                p_box4_P.setAutoDraw(True)
            
            # if p_box4_P is active this frame...
            if p_box4_P.status == STARTED:
                # update params
                pass
            
            # *p_choice* updates
            waitOnFlip = False
            
            # if p_choice is starting this frame...
            if p_choice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_choice.frameNStart = frameN  # exact frame index
                p_choice.tStart = t  # local t and not account for scr refresh
                p_choice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_choice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_choice.started')
                # update status
                p_choice.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(p_choice.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(p_choice.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if p_choice.status == STARTED and not waitOnFlip:
                theseKeys = p_choice.getKeys(keyList=['left', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _p_choice_allKeys.extend(theseKeys)
                if len(_p_choice_allKeys):
                    p_choice.keys = _p_choice_allKeys[-1].name  # just the last key pressed
                    p_choice.rt = _p_choice_allKeys[-1].rt
                    p_choice.duration = _p_choice_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                p_options_show.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in p_options_show.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "p_options_show" ---
        for thisComponent in p_options_show.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for p_options_show
        p_options_show.tStop = globalClock.getTime(format='float')
        p_options_show.tStopRefresh = tThisFlipGlobal
        thisExp.addData('p_options_show.stopped', p_options_show.tStop)
        # check responses
        if p_choice.keys in ['', [], None]:  # No response was made
            p_choice.keys = None
        trials.addData('p_choice.keys',p_choice.keys)
        if p_choice.keys != None:  # we had a response
            trials.addData('p_choice.rt', p_choice.rt)
            trials.addData('p_choice.duration', p_choice.duration)
        # the Routine "p_options_show" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "gamble_choice" ---
        # create an object to store info about Routine gamble_choice
        gamble_choice = data.Routine(
            name='gamble_choice',
            components=[fixation_cross_3, gamble1_rew_test_2, gamble1_rew_text_low_2, gamble1_prob_text_2, gamble2_rew_text_2, gamble2_rew_text_low_2, gamble2_prob_text_2, gamblechoice, Choose_text],
        )
        gamble_choice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        gamble1_rew_test_2.setText(gambles_txt[trials.thisN][1])
        gamble1_rew_text_low_2.setText(gambles_txt[trials.thisN][2])
        gamble1_prob_text_2.setText(gambles_txt[trials.thisN][0])
        gamble2_rew_text_2.setText(gambles_txt[trials.thisN][4])
        gamble2_rew_text_low_2.setText(gambles_txt[trials.thisN][5])
        gamble2_prob_text_2.setText(gambles_txt[trials.thisN][3])
        # create starting attributes for gamblechoice
        gamblechoice.keys = []
        gamblechoice.rt = []
        _gamblechoice_allKeys = []
        # store start times for gamble_choice
        gamble_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        gamble_choice.tStart = globalClock.getTime(format='float')
        gamble_choice.status = STARTED
        thisExp.addData('gamble_choice.started', gamble_choice.tStart)
        gamble_choice.maxDuration = None
        # keep track of which components have finished
        gamble_choiceComponents = gamble_choice.components
        for thisComponent in gamble_choice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "gamble_choice" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        gamble_choice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from gamble_choice_code
            Gamble1_green.draw()
            Gamble1_red.draw()
            Gamble2_green.draw()
            Gamble2_red.draw()
            
            # *fixation_cross_3* updates
            
            # if fixation_cross_3 is starting this frame...
            if fixation_cross_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross_3.frameNStart = frameN  # exact frame index
                fixation_cross_3.tStart = t  # local t and not account for scr refresh
                fixation_cross_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross_3.started')
                # update status
                fixation_cross_3.status = STARTED
                fixation_cross_3.setAutoDraw(True)
            
            # if fixation_cross_3 is active this frame...
            if fixation_cross_3.status == STARTED:
                # update params
                pass
            
            # *gamble1_rew_test_2* updates
            
            # if gamble1_rew_test_2 is starting this frame...
            if gamble1_rew_test_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamble1_rew_test_2.frameNStart = frameN  # exact frame index
                gamble1_rew_test_2.tStart = t  # local t and not account for scr refresh
                gamble1_rew_test_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble1_rew_test_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble1_rew_test_2.started')
                # update status
                gamble1_rew_test_2.status = STARTED
                gamble1_rew_test_2.setAutoDraw(True)
            
            # if gamble1_rew_test_2 is active this frame...
            if gamble1_rew_test_2.status == STARTED:
                # update params
                pass
            
            # *gamble1_rew_text_low_2* updates
            
            # if gamble1_rew_text_low_2 is starting this frame...
            if gamble1_rew_text_low_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamble1_rew_text_low_2.frameNStart = frameN  # exact frame index
                gamble1_rew_text_low_2.tStart = t  # local t and not account for scr refresh
                gamble1_rew_text_low_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble1_rew_text_low_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble1_rew_text_low_2.started')
                # update status
                gamble1_rew_text_low_2.status = STARTED
                gamble1_rew_text_low_2.setAutoDraw(True)
            
            # if gamble1_rew_text_low_2 is active this frame...
            if gamble1_rew_text_low_2.status == STARTED:
                # update params
                pass
            
            # *gamble1_prob_text_2* updates
            
            # if gamble1_prob_text_2 is starting this frame...
            if gamble1_prob_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamble1_prob_text_2.frameNStart = frameN  # exact frame index
                gamble1_prob_text_2.tStart = t  # local t and not account for scr refresh
                gamble1_prob_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble1_prob_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble1_prob_text_2.started')
                # update status
                gamble1_prob_text_2.status = STARTED
                gamble1_prob_text_2.setAutoDraw(True)
            
            # if gamble1_prob_text_2 is active this frame...
            if gamble1_prob_text_2.status == STARTED:
                # update params
                pass
            
            # *gamble2_rew_text_2* updates
            
            # if gamble2_rew_text_2 is starting this frame...
            if gamble2_rew_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamble2_rew_text_2.frameNStart = frameN  # exact frame index
                gamble2_rew_text_2.tStart = t  # local t and not account for scr refresh
                gamble2_rew_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble2_rew_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble2_rew_text_2.started')
                # update status
                gamble2_rew_text_2.status = STARTED
                gamble2_rew_text_2.setAutoDraw(True)
            
            # if gamble2_rew_text_2 is active this frame...
            if gamble2_rew_text_2.status == STARTED:
                # update params
                pass
            
            # *gamble2_rew_text_low_2* updates
            
            # if gamble2_rew_text_low_2 is starting this frame...
            if gamble2_rew_text_low_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamble2_rew_text_low_2.frameNStart = frameN  # exact frame index
                gamble2_rew_text_low_2.tStart = t  # local t and not account for scr refresh
                gamble2_rew_text_low_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble2_rew_text_low_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble2_rew_text_low_2.started')
                # update status
                gamble2_rew_text_low_2.status = STARTED
                gamble2_rew_text_low_2.setAutoDraw(True)
            
            # if gamble2_rew_text_low_2 is active this frame...
            if gamble2_rew_text_low_2.status == STARTED:
                # update params
                pass
            
            # *gamble2_prob_text_2* updates
            
            # if gamble2_prob_text_2 is starting this frame...
            if gamble2_prob_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamble2_prob_text_2.frameNStart = frameN  # exact frame index
                gamble2_prob_text_2.tStart = t  # local t and not account for scr refresh
                gamble2_prob_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamble2_prob_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamble2_prob_text_2.started')
                # update status
                gamble2_prob_text_2.status = STARTED
                gamble2_prob_text_2.setAutoDraw(True)
            
            # if gamble2_prob_text_2 is active this frame...
            if gamble2_prob_text_2.status == STARTED:
                # update params
                pass
            
            # *gamblechoice* updates
            waitOnFlip = False
            
            # if gamblechoice is starting this frame...
            if gamblechoice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gamblechoice.frameNStart = frameN  # exact frame index
                gamblechoice.tStart = t  # local t and not account for scr refresh
                gamblechoice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gamblechoice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gamblechoice.started')
                # update status
                gamblechoice.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(gamblechoice.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(gamblechoice.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if gamblechoice.status == STARTED and not waitOnFlip:
                theseKeys = gamblechoice.getKeys(keyList=['left', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _gamblechoice_allKeys.extend(theseKeys)
                if len(_gamblechoice_allKeys):
                    gamblechoice.keys = _gamblechoice_allKeys[-1].name  # just the last key pressed
                    gamblechoice.rt = _gamblechoice_allKeys[-1].rt
                    gamblechoice.duration = _gamblechoice_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *Choose_text* updates
            
            # if Choose_text is starting this frame...
            if Choose_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Choose_text.frameNStart = frameN  # exact frame index
                Choose_text.tStart = t  # local t and not account for scr refresh
                Choose_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Choose_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Choose_text.started')
                # update status
                Choose_text.status = STARTED
                Choose_text.setAutoDraw(True)
            
            # if Choose_text is active this frame...
            if Choose_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                gamble_choice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in gamble_choice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "gamble_choice" ---
        for thisComponent in gamble_choice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for gamble_choice
        gamble_choice.tStop = globalClock.getTime(format='float')
        gamble_choice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('gamble_choice.stopped', gamble_choice.tStop)
        # check responses
        if gamblechoice.keys in ['', [], None]:  # No response was made
            gamblechoice.keys = None
        trials.addData('gamblechoice.keys',gamblechoice.keys)
        if gamblechoice.keys != None:  # we had a response
            trials.addData('gamblechoice.rt', gamblechoice.rt)
            trials.addData('gamblechoice.duration', gamblechoice.duration)
        # the Routine "gamble_choice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "chosen_gamble" ---
        # create an object to store info about Routine chosen_gamble
        chosen_gamble = data.Routine(
            name='chosen_gamble',
            components=[fixation_7, chosen_reward_txt, chosen_reward_txt_low, chosen_prob_txt],
        )
        chosen_gamble.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from chosen_gamble_code
        inputs = event.getKeys()
        userchoice = inputs[-1]
        # the two preceding lines of code do not function properly on a M4 macbook
        
        rightn = trials.thisN + (npractice_trials*3)
        if userchoice == 'left':
            choice_pos = -width
            rewa_txt = gambles_txt[rightn][1]
            prob_txt = gambles_txt[rightn][0]
            gamble_size = gamble1_size # Will be used for the outcome square size (next rountine)
            choice = 0
        else:
            choice_pos = width
            rewa_txt = gambles_txt[rightn][3]
            prob_txt = gambles_txt[rightn][2]
            gamble_size = gamble2_size # Will be used for the outcome square size (next rountine)
            choice = 1
        thisExp.addData('choice', choice)
        chosen_reward_txt.setPos((choice_pos, reward[2]+rew_txt_margin+0.03))
        chosen_reward_txt.setText(rewa_txt)
        chosen_reward_txt_low.setPos((choice_pos, reward[2]+rew_txt_margin))
        chosen_reward_txt_low.setText(rewa_txt_low)
        chosen_prob_txt.setPos((choice_pos, 0))
        chosen_prob_txt.setText(prob_txt)
        # store start times for chosen_gamble
        chosen_gamble.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        chosen_gamble.tStart = globalClock.getTime(format='float')
        chosen_gamble.status = STARTED
        thisExp.addData('chosen_gamble.started', chosen_gamble.tStart)
        chosen_gamble.maxDuration = time_chosen_gamble
        # keep track of which components have finished
        chosen_gambleComponents = chosen_gamble.components
        for thisComponent in chosen_gamble.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "chosen_gamble" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        chosen_gamble.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > chosen_gamble.maxDuration-frameTolerance:
                chosen_gamble.maxDurationReached = True
                continueRoutine = False
            # Run 'Each Frame' code from chosen_gamble_code
            if userchoice == 'left':
                Gamble1_green.draw()
                Gamble1_red.draw()
            else:
                Gamble2_green.draw()
                Gamble2_red.draw()
            
            # *fixation_7* updates
            
            # if fixation_7 is starting this frame...
            if fixation_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_7.frameNStart = frameN  # exact frame index
                fixation_7.tStart = t  # local t and not account for scr refresh
                fixation_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_7.started')
                # update status
                fixation_7.status = STARTED
                fixation_7.setAutoDraw(True)
            
            # if fixation_7 is active this frame...
            if fixation_7.status == STARTED:
                # update params
                pass
            
            # *chosen_reward_txt* updates
            
            # if chosen_reward_txt is starting this frame...
            if chosen_reward_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                chosen_reward_txt.frameNStart = frameN  # exact frame index
                chosen_reward_txt.tStart = t  # local t and not account for scr refresh
                chosen_reward_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(chosen_reward_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'chosen_reward_txt.started')
                # update status
                chosen_reward_txt.status = STARTED
                chosen_reward_txt.setAutoDraw(True)
            
            # if chosen_reward_txt is active this frame...
            if chosen_reward_txt.status == STARTED:
                # update params
                pass
            
            # *chosen_reward_txt_low* updates
            
            # if chosen_reward_txt_low is starting this frame...
            if chosen_reward_txt_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                chosen_reward_txt_low.frameNStart = frameN  # exact frame index
                chosen_reward_txt_low.tStart = t  # local t and not account for scr refresh
                chosen_reward_txt_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(chosen_reward_txt_low, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'chosen_reward_txt_low.started')
                # update status
                chosen_reward_txt_low.status = STARTED
                chosen_reward_txt_low.setAutoDraw(True)
            
            # if chosen_reward_txt_low is active this frame...
            if chosen_reward_txt_low.status == STARTED:
                # update params
                pass
            
            # *chosen_prob_txt* updates
            
            # if chosen_prob_txt is starting this frame...
            if chosen_prob_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                chosen_prob_txt.frameNStart = frameN  # exact frame index
                chosen_prob_txt.tStart = t  # local t and not account for scr refresh
                chosen_prob_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(chosen_prob_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'chosen_prob_txt.started')
                # update status
                chosen_prob_txt.status = STARTED
                chosen_prob_txt.setAutoDraw(True)
            
            # if chosen_prob_txt is active this frame...
            if chosen_prob_txt.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                chosen_gamble.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in chosen_gamble.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "chosen_gamble" ---
        for thisComponent in chosen_gamble.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for chosen_gamble
        chosen_gamble.tStop = globalClock.getTime(format='float')
        chosen_gamble.tStopRefresh = tThisFlipGlobal
        thisExp.addData('chosen_gamble.stopped', chosen_gamble.tStop)
        # the Routine "chosen_gamble" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "gamble_outcome" ---
        # create an object to store info about Routine gamble_outcome
        gamble_outcome = data.Routine(
            name='gamble_outcome',
            components=[fixation_8, money_prompt, current_money_txt, outcome_square, outcome_text_3, outcome_text_2, prog_bar, next_trial_txt, next_trial_input],
        )
        gamble_outcome.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from gamble_result_code
        progBar = progVal/ntrials
        progVal += 1
        
        event.clearEvents()
        
        if userchoice == 'left':
            if gambles[trials.thisN][0] >= gamble_result[trials.thisN]:
                current_money += gambles[trials.thisN][1]
                outcome_txtW = 'Win!'
                W_pos = 0.03
                outcome_txtN = gambles_txt[trials.thisN][1]
                N_pos = -0.03
                outcome_color = 'Green'
                outcome = 1
            else:
                outcome_txtW = 'Loss!'
                W_pos = 0
                outcome_txtN = ''
                N_pos = -1.5
                outcome_color = 'Red'
                outcome = 0
        else:
            if gambles[trials.thisN][2] >= gamble_result[trials.thisN]:
                current_money += gambles[trials.thisN][3]
                outcome_txtW = 'Win!'
                W_pos = 0.03
                outcome_txtN = gambles_txt[trials.thisN][3]
                N_pos = -0.03
                outcome_color = 'Green'
                outcome = 1
            else:
                outcome_txtW = 'Loss!'
                W_pos = 0
                outcome_txtN = ''
                N_pos = -1.5
                outcome_color = 'Red'
                outcome = 0
        thisExp.addData('current_money', current_money)
        thisExp.addData("outcome", outcome)
        current_money_txt.setText(money_txt)
        outcome_square.setFillColor(outcome_color)
        outcome_square.setPos((choice_pos, 0))
        outcome_square.setSize((2*gamble_size, 2*gamble_size))
        outcome_square.setLineColor(outcome_color)
        outcome_text_3.setPos((choice_pos, W_pos))
        outcome_text_3.setText(outcome_txtW)
        outcome_text_2.setPos((choice_pos, rewa_pos))
        outcome_text_2.setText(outcome_rewa + '\n' + outcome_rewaL_txt)
        prog_bar.setProgress(progBar)
        # create starting attributes for next_trial_input
        next_trial_input.keys = []
        next_trial_input.rt = []
        _next_trial_input_allKeys = []
        # store start times for gamble_outcome
        gamble_outcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        gamble_outcome.tStart = globalClock.getTime(format='float')
        gamble_outcome.status = STARTED
        thisExp.addData('gamble_outcome.started', gamble_outcome.tStart)
        gamble_outcome.maxDuration = None
        # keep track of which components have finished
        gamble_outcomeComponents = gamble_outcome.components
        for thisComponent in gamble_outcome.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "gamble_outcome" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        gamble_outcome.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_8* updates
            
            # if fixation_8 is starting this frame...
            if fixation_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_8.frameNStart = frameN  # exact frame index
                fixation_8.tStart = t  # local t and not account for scr refresh
                fixation_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_8.started')
                # update status
                fixation_8.status = STARTED
                fixation_8.setAutoDraw(True)
            
            # if fixation_8 is active this frame...
            if fixation_8.status == STARTED:
                # update params
                pass
            
            # *money_prompt* updates
            
            # if money_prompt is starting this frame...
            if money_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                money_prompt.frameNStart = frameN  # exact frame index
                money_prompt.tStart = t  # local t and not account for scr refresh
                money_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(money_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'money_prompt.started')
                # update status
                money_prompt.status = STARTED
                money_prompt.setAutoDraw(True)
            
            # if money_prompt is active this frame...
            if money_prompt.status == STARTED:
                # update params
                pass
            
            # *current_money_txt* updates
            
            # if current_money_txt is starting this frame...
            if current_money_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                current_money_txt.frameNStart = frameN  # exact frame index
                current_money_txt.tStart = t  # local t and not account for scr refresh
                current_money_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(current_money_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'current_money_txt.started')
                # update status
                current_money_txt.status = STARTED
                current_money_txt.setAutoDraw(True)
            
            # if current_money_txt is active this frame...
            if current_money_txt.status == STARTED:
                # update params
                pass
            
            # *outcome_square* updates
            
            # if outcome_square is starting this frame...
            if outcome_square.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                outcome_square.frameNStart = frameN  # exact frame index
                outcome_square.tStart = t  # local t and not account for scr refresh
                outcome_square.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(outcome_square, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'outcome_square.started')
                # update status
                outcome_square.status = STARTED
                outcome_square.setAutoDraw(True)
            
            # if outcome_square is active this frame...
            if outcome_square.status == STARTED:
                # update params
                pass
            
            # *outcome_text_3* updates
            
            # if outcome_text_3 is starting this frame...
            if outcome_text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                outcome_text_3.frameNStart = frameN  # exact frame index
                outcome_text_3.tStart = t  # local t and not account for scr refresh
                outcome_text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(outcome_text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'outcome_text_3.started')
                # update status
                outcome_text_3.status = STARTED
                outcome_text_3.setAutoDraw(True)
            
            # if outcome_text_3 is active this frame...
            if outcome_text_3.status == STARTED:
                # update params
                pass
            
            # *outcome_text_2* updates
            
            # if outcome_text_2 is starting this frame...
            if outcome_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                outcome_text_2.frameNStart = frameN  # exact frame index
                outcome_text_2.tStart = t  # local t and not account for scr refresh
                outcome_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(outcome_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'outcome_text_2.started')
                # update status
                outcome_text_2.status = STARTED
                outcome_text_2.setAutoDraw(True)
            
            # if outcome_text_2 is active this frame...
            if outcome_text_2.status == STARTED:
                # update params
                pass
            
            # *prog_bar* updates
            
            # if prog_bar is starting this frame...
            if prog_bar.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog_bar.frameNStart = frameN  # exact frame index
                prog_bar.tStart = t  # local t and not account for scr refresh
                prog_bar.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog_bar, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prog_bar.started')
                # update status
                prog_bar.status = STARTED
                prog_bar.setAutoDraw(True)
            
            # if prog_bar is active this frame...
            if prog_bar.status == STARTED:
                # update params
                pass
            
            # *next_trial_txt* updates
            
            # if next_trial_txt is starting this frame...
            if next_trial_txt.status == NOT_STARTED and tThisFlip >= time_gamble_result-frameTolerance:
                # keep track of start time/frame for later
                next_trial_txt.frameNStart = frameN  # exact frame index
                next_trial_txt.tStart = t  # local t and not account for scr refresh
                next_trial_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(next_trial_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'next_trial_txt.started')
                # update status
                next_trial_txt.status = STARTED
                next_trial_txt.setAutoDraw(True)
            
            # if next_trial_txt is active this frame...
            if next_trial_txt.status == STARTED:
                # update params
                pass
            
            # *next_trial_input* updates
            waitOnFlip = False
            
            # if next_trial_input is starting this frame...
            if next_trial_input.status == NOT_STARTED and tThisFlip >= time_gamble_result-frameTolerance:
                # keep track of start time/frame for later
                next_trial_input.frameNStart = frameN  # exact frame index
                next_trial_input.tStart = t  # local t and not account for scr refresh
                next_trial_input.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(next_trial_input, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'next_trial_input.started')
                # update status
                next_trial_input.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(next_trial_input.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(next_trial_input.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if next_trial_input.status == STARTED and not waitOnFlip:
                theseKeys = next_trial_input.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _next_trial_input_allKeys.extend(theseKeys)
                if len(_next_trial_input_allKeys):
                    next_trial_input.keys = _next_trial_input_allKeys[-1].name  # just the last key pressed
                    next_trial_input.rt = _next_trial_input_allKeys[-1].rt
                    next_trial_input.duration = _next_trial_input_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                gamble_outcome.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in gamble_outcome.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "gamble_outcome" ---
        for thisComponent in gamble_outcome.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for gamble_outcome
        gamble_outcome.tStop = globalClock.getTime(format='float')
        gamble_outcome.tStopRefresh = tThisFlipGlobal
        thisExp.addData('gamble_outcome.stopped', gamble_outcome.tStop)
        # check responses
        if next_trial_input.keys in ['', [], None]:  # No response was made
            next_trial_input.keys = None
        trials.addData('next_trial_input.keys',next_trial_input.keys)
        if next_trial_input.keys != None:  # we had a response
            trials.addData('next_trial_input.rt', next_trial_input.rt)
            trials.addData('next_trial_input.duration', next_trial_input.duration)
        # the Routine "gamble_outcome" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "iti" ---
        # create an object to store info about Routine iti
        iti = data.Routine(
            name='iti',
            components=[text],
        )
        iti.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for iti
        iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti.tStart = globalClock.getTime(format='float')
        iti.status = STARTED
        thisExp.addData('iti.started', iti.tStart)
        iti.maxDuration = time_iti
        # keep track of which components have finished
        itiComponents = iti.components
        for thisComponent in iti.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        iti.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > iti.maxDuration-frameTolerance:
                iti.maxDurationReached = True
                continueRoutine = False
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti" ---
        for thisComponent in iti.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti
        iti.tStop = globalClock.getTime(format='float')
        iti.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti.stopped', iti.tStop)
        # the Routine "iti" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "happiness_trial" ---
        # create an object to store info about Routine happiness_trial
        happiness_trial = data.Routine(
            name='happiness_trial',
            components=[happiness_rating, low_end_text_2, high_end_text_2, question_2, exit_text, key_resp_6],
        )
        happiness_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        happiness_rating.reset()
        # create starting attributes for key_resp_6
        key_resp_6.keys = []
        key_resp_6.rt = []
        _key_resp_6_allKeys = []
        # store start times for happiness_trial
        happiness_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        happiness_trial.tStart = globalClock.getTime(format='float')
        happiness_trial.status = STARTED
        thisExp.addData('happiness_trial.started', happiness_trial.tStart)
        happiness_trial.maxDuration = None
        # skip Routine happiness_trial if its 'Skip if' condition is True
        happiness_trial.skipped = continueRoutine and not (not (happyTrial[trials.thisN]))
        continueRoutine = happiness_trial.skipped
        # keep track of which components have finished
        happiness_trialComponents = happiness_trial.components
        for thisComponent in happiness_trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "happiness_trial" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        happiness_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *happiness_rating* updates
            
            # if happiness_rating is starting this frame...
            if happiness_rating.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                happiness_rating.frameNStart = frameN  # exact frame index
                happiness_rating.tStart = t  # local t and not account for scr refresh
                happiness_rating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(happiness_rating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'happiness_rating.started')
                # update status
                happiness_rating.status = STARTED
                happiness_rating.setAutoDraw(True)
            
            # if happiness_rating is active this frame...
            if happiness_rating.status == STARTED:
                # update params
                pass
            
            # *low_end_text_2* updates
            
            # if low_end_text_2 is starting this frame...
            if low_end_text_2.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                low_end_text_2.frameNStart = frameN  # exact frame index
                low_end_text_2.tStart = t  # local t and not account for scr refresh
                low_end_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(low_end_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'low_end_text_2.started')
                # update status
                low_end_text_2.status = STARTED
                low_end_text_2.setAutoDraw(True)
            
            # if low_end_text_2 is active this frame...
            if low_end_text_2.status == STARTED:
                # update params
                pass
            
            # *high_end_text_2* updates
            
            # if high_end_text_2 is starting this frame...
            if high_end_text_2.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                high_end_text_2.frameNStart = frameN  # exact frame index
                high_end_text_2.tStart = t  # local t and not account for scr refresh
                high_end_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(high_end_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'high_end_text_2.started')
                # update status
                high_end_text_2.status = STARTED
                high_end_text_2.setAutoDraw(True)
            
            # if high_end_text_2 is active this frame...
            if high_end_text_2.status == STARTED:
                # update params
                pass
            
            # *question_2* updates
            
            # if question_2 is starting this frame...
            if question_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_2.frameNStart = frameN  # exact frame index
                question_2.tStart = t  # local t and not account for scr refresh
                question_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question_2.started')
                # update status
                question_2.status = STARTED
                question_2.setAutoDraw(True)
            
            # if question_2 is active this frame...
            if question_2.status == STARTED:
                # update params
                pass
            
            # *exit_text* updates
            
            # if exit_text is starting this frame...
            if exit_text.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                exit_text.frameNStart = frameN  # exact frame index
                exit_text.tStart = t  # local t and not account for scr refresh
                exit_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(exit_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'exit_text.started')
                # update status
                exit_text.status = STARTED
                exit_text.setAutoDraw(True)
            
            # if exit_text is active this frame...
            if exit_text.status == STARTED:
                # update params
                pass
            
            # *key_resp_6* updates
            waitOnFlip = False
            
            # if key_resp_6 is starting this frame...
            if key_resp_6.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_6.frameNStart = frameN  # exact frame index
                key_resp_6.tStart = t  # local t and not account for scr refresh
                key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_6.started')
                # update status
                key_resp_6.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_6.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_6.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_6_allKeys.extend(theseKeys)
                if len(_key_resp_6_allKeys):
                    key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                    key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                    key_resp_6.duration = _key_resp_6_allKeys[-1].duration
            # Run 'Each Frame' code from code_8
            if happiness_rating_overall.getRating() != undefined  and events.getKeys() == "return" :
                continueRoutine = false
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                happiness_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in happiness_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "happiness_trial" ---
        for thisComponent in happiness_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for happiness_trial
        happiness_trial.tStop = globalClock.getTime(format='float')
        happiness_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('happiness_trial.stopped', happiness_trial.tStop)
        trials.addData('happiness_rating.response', happiness_rating.getRating())
        trials.addData('happiness_rating.rt', happiness_rating.getRT())
        # check responses
        if key_resp_6.keys in ['', [], None]:  # No response was made
            key_resp_6.keys = None
        trials.addData('key_resp_6.keys',key_resp_6.keys)
        if key_resp_6.keys != None:  # we had a response
            trials.addData('key_resp_6.rt', key_resp_6.rt)
            trials.addData('key_resp_6.duration', key_resp_6.duration)
        # the Routine "happiness_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "h_iti" ---
        # create an object to store info about Routine h_iti
        h_iti = data.Routine(
            name='h_iti',
            components=[text_7],
        )
        h_iti.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for h_iti
        h_iti.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        h_iti.tStart = globalClock.getTime(format='float')
        h_iti.status = STARTED
        thisExp.addData('h_iti.started', h_iti.tStart)
        h_iti.maxDuration = time_iti
        # skip Routine h_iti if its 'Skip if' condition is True
        h_iti.skipped = continueRoutine and not (not (happyTrial[trials.thisN]))
        continueRoutine = h_iti.skipped
        # keep track of which components have finished
        h_itiComponents = h_iti.components
        for thisComponent in h_iti.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "h_iti" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        h_iti.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > h_iti.maxDuration-frameTolerance:
                h_iti.maxDurationReached = True
                continueRoutine = False
            
            # *text_7* updates
            
            # if text_7 is starting this frame...
            if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_7.frameNStart = frameN  # exact frame index
                text_7.tStart = t  # local t and not account for scr refresh
                text_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_7.started')
                # update status
                text_7.status = STARTED
                text_7.setAutoDraw(True)
            
            # if text_7 is active this frame...
            if text_7.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                h_iti.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in h_iti.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "h_iti" ---
        for thisComponent in h_iti.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for h_iti
        h_iti.tStop = globalClock.getTime(format='float')
        h_iti.tStopRefresh = tThisFlipGlobal
        thisExp.addData('h_iti.stopped', h_iti.tStop)
        # the Routine "h_iti" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_break" ---
        # create an object to store info about Routine trial_break
        trial_break = data.Routine(
            name='trial_break',
            components=[break_txt, key_resp_10],
        )
        trial_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_10
        key_resp_10.keys = []
        key_resp_10.rt = []
        _key_resp_10_allKeys = []
        # store start times for trial_break
        trial_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_break.tStart = globalClock.getTime(format='float')
        trial_break.status = STARTED
        thisExp.addData('trial_break.started', trial_break.tStart)
        trial_break.maxDuration = None
        # skip Routine trial_break if its 'Skip if' condition is True
        trial_break.skipped = continueRoutine and not ((trials.thisN % 37 != 0) | (trials.thisN != 148))
        continueRoutine = trial_break.skipped
        # keep track of which components have finished
        trial_breakComponents = trial_break.components
        for thisComponent in trial_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_break" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        trial_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_txt* updates
            
            # if break_txt is starting this frame...
            if break_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_txt.frameNStart = frameN  # exact frame index
                break_txt.tStart = t  # local t and not account for scr refresh
                break_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_txt.started')
                # update status
                break_txt.status = STARTED
                break_txt.setAutoDraw(True)
            
            # if break_txt is active this frame...
            if break_txt.status == STARTED:
                # update params
                pass
            
            # *key_resp_10* updates
            waitOnFlip = False
            
            # if key_resp_10 is starting this frame...
            if key_resp_10.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_10.frameNStart = frameN  # exact frame index
                key_resp_10.tStart = t  # local t and not account for scr refresh
                key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_10.started')
                # update status
                key_resp_10.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_10.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_10.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_10_allKeys.extend(theseKeys)
                if len(_key_resp_10_allKeys):
                    key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                    key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                    key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_break" ---
        for thisComponent in trial_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_break
        trial_break.tStop = globalClock.getTime(format='float')
        trial_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_break.stopped', trial_break.tStop)
        # check responses
        if key_resp_10.keys in ['', [], None]:  # No response was made
            key_resp_10.keys = None
        trials.addData('key_resp_10.keys',key_resp_10.keys)
        if key_resp_10.keys != None:  # we had a response
            trials.addData('key_resp_10.rt', key_resp_10.rt)
            trials.addData('key_resp_10.duration', key_resp_10.duration)
        # the Routine "trial_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "repeat_iti_2" ---
        # create an object to store info about Routine repeat_iti_2
        repeat_iti_2 = data.Routine(
            name='repeat_iti_2',
            components=[text_6],
        )
        repeat_iti_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for repeat_iti_2
        repeat_iti_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        repeat_iti_2.tStart = globalClock.getTime(format='float')
        repeat_iti_2.status = STARTED
        thisExp.addData('repeat_iti_2.started', repeat_iti_2.tStart)
        repeat_iti_2.maxDuration = time_iti
        # skip Routine repeat_iti_2 if its 'Skip if' condition is True
        repeat_iti_2.skipped = continueRoutine and not ((trials.thisN % 37 != 0) | (trials.thisN != 148))
        continueRoutine = repeat_iti_2.skipped
        # keep track of which components have finished
        repeat_iti_2Components = repeat_iti_2.components
        for thisComponent in repeat_iti_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "repeat_iti_2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        repeat_iti_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > repeat_iti_2.maxDuration-frameTolerance:
                repeat_iti_2.maxDurationReached = True
                continueRoutine = False
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_6.started')
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                repeat_iti_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in repeat_iti_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "repeat_iti_2" ---
        for thisComponent in repeat_iti_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for repeat_iti_2
        repeat_iti_2.tStop = globalClock.getTime(format='float')
        repeat_iti_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('repeat_iti_2.stopped', repeat_iti_2.tStop)
        # the Routine "repeat_iti_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed ntrials repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Thank_you" ---
    # create an object to store info about Routine Thank_you
    Thank_you = data.Routine(
        name='Thank_you',
        components=[thankYouText, key_resp_11],
    )
    Thank_you.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_11
    key_resp_11.keys = []
    key_resp_11.rt = []
    _key_resp_11_allKeys = []
    # store start times for Thank_you
    Thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Thank_you.tStart = globalClock.getTime(format='float')
    Thank_you.status = STARTED
    thisExp.addData('Thank_you.started', Thank_you.tStart)
    Thank_you.maxDuration = None
    # keep track of which components have finished
    Thank_youComponents = Thank_you.components
    for thisComponent in Thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Thank_you" ---
    Thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thankYouText* updates
        
        # if thankYouText is starting this frame...
        if thankYouText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thankYouText.frameNStart = frameN  # exact frame index
            thankYouText.tStart = t  # local t and not account for scr refresh
            thankYouText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thankYouText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thankYouText.started')
            # update status
            thankYouText.status = STARTED
            thankYouText.setAutoDraw(True)
        
        # if thankYouText is active this frame...
        if thankYouText.status == STARTED:
            # update params
            pass
        
        # *key_resp_11* updates
        waitOnFlip = False
        
        # if key_resp_11 is starting this frame...
        if key_resp_11.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            key_resp_11.frameNStart = frameN  # exact frame index
            key_resp_11.tStart = t  # local t and not account for scr refresh
            key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_11.started')
            # update status
            key_resp_11.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_11.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_11.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_11_allKeys.extend(theseKeys)
            if len(_key_resp_11_allKeys):
                key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Thank_you.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Thank_you" ---
    for thisComponent in Thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Thank_you
    Thank_you.tStop = globalClock.getTime(format='float')
    Thank_you.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Thank_you.stopped', Thank_you.tStop)
    # check responses
    if key_resp_11.keys in ['', [], None]:  # No response was made
        key_resp_11.keys = None
    thisExp.addData('key_resp_11.keys',key_resp_11.keys)
    if key_resp_11.keys != None:  # we had a response
        thisExp.addData('key_resp_11.rt', key_resp_11.rt)
        thisExp.addData('key_resp_11.duration', key_resp_11.duration)
    thisExp.nextEntry()
    # the Routine "Thank_you" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
