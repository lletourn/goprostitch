#!/bin/env python3
import argparse
import bs4
import dataclasses
from dateutil.parser import parse
import datetime
import json
import logging
import pytz  # type: ignore
import re
import requests
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Player:
    jersey: str
    first_name: str
    last_name: str


@dataclasses.dataclass(frozen=True)
class Game:
    scheduled_time: datetime.datetime
    venue: str
    home_team_name: str
    away_team_name: str
    home_players: List[Player]
    away_players: List[Player]


def parse_players(tbody: bs4.element.Tag) -> List[Player]:
    players: List[Player] = list()
    for tr in tbody.find_all('tr'):
        tds = tr.find_all('td')

        jersey = tds[0].string
        full_name = tds[1].a.get_text()
        name = full_name.split(' ')
        first_name = name[0]
        last_name = ' '.join(name[1:])

        players.append(Player(jersey=jersey, first_name=first_name, last_name=last_name))

    return players


def parse_boxscore(game_id: int) -> Game:
    boxscore_url = f"https://www.lhdrs.com/stats/scoresheet/{game_id}"
    logger.debug("Boxscore url: %s", boxscore_url)

    resp = requests.get(boxscore_url)
    soup = bs4.BeautifulSoup(resp.text, 'html.parser')
    # print(soup.prettify())

    teams = soup.find_all("h3", class_="team-title")
    home_team_name = str(teams[0].string)
    away_team_name = str(teams[1].string)

    players = soup.find_all("table", class_="players-team")
    home_players = parse_players(players[0].tbody)
    away_players = parse_players(players[1].tbody)

    scorepanel = soup.find(id='scorepanel')
    assert scorepanel is not None, "Scorepanel doesn't exist"

    eastern = pytz.timezone('America/Montreal')
    pattern = re.compile('.*Date : (.*)Ar.na : (.*)', re.DOTALL)
    sibling = scorepanel.next_sibling

    tz_aware_date: Optional[datetime.datetime] = None
    while sibling:
        txt = sibling.get_text()
        if 'Date' in txt:
            m = pattern.match(txt)
            if m:
                tz_naive_date = parse(m.group(1).strip())
                tz_aware_date = eastern.localize(tz_naive_date, is_dst=None)
                venue = m.group(2).strip()
                break
        sibling = sibling.next_sibling

    if tz_aware_date is None:
        raise ValueError("Missing game scheduled time")
    game = Game(scheduled_time=tz_aware_date, venue=venue, home_team_name=home_team_name, away_team_name=away_team_name, home_players=home_players, away_players=away_players)

    return game


def build_request(game: Game, game_start: int, game_end: int, left_goalie_team: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    players: List[Dict[str, Any]] = list()
    game_request: Dict[str, Any] = {"requestType": "processGame", "requestDetails": {"surface": {"surfaceRefId": game.venue}}}

    game_request["requestDetails"]["media"] = [{"mediaRefId": "A", "url": "https://slextractedclips.s3.amazonaws.com/"}]
    game_request["requestDetails"]["game"] = {"gameRefId": f"LHDRS_{game.scheduled_time}", "scheduledTime": f"{game.scheduled_time.astimezone(pytz.utc).isoformat()}"}
    game_request["requestDetails"]["game"]["home"] = {"teamRefId": game.home_team_name, "name": game.home_team_name}
    game_request["requestDetails"]["game"]["away"] = {"teamRefId": game.away_team_name, "name": game.away_team_name}
    game_request["requestDetails"]["game"]["league"] = {"leagueRefId": "LHDRS", "name": "LHDRS"}

    game_request["requestDetails"]["timestamps"] = []

    left_side_goalie_team = game.away_team_name
    if left_goalie_team == 'home':
        left_side_goalie_team = game.home_team_name

    game_request["requestDetails"]["timestamps"].append({"mediaRefId": "A", "videoTimeMS": float(game_start) * 1000.0, "type": "gameStart", "period": 1, "leftSideGoalTeam": left_side_goalie_team})
    game_request["requestDetails"]["timestamps"].append({"mediaRefId": "A", "videoTimeMS": float(game_end) * 1000.0, "type": "gameEnd", "period": 3, "leftSideGoalTeam": left_side_goalie_team})

    game_request["requestDetails"]["rosters"] = {"teams": [{"teamRefId": game.home_team_name, "players": []}, {"teamRefId": game.away_team_name, "players": []}]}

    if left_goalie_team == 'home':
        home_goalie_side = 'left'
        away_goalie_side = 'right'
    else:
        home_goalie_side = 'right'
        away_goalie_side = 'left'

    for player in game.home_players:
        game_request["requestDetails"]["rosters"]["teams"][0]["players"].append({"playerRefId": f"H{player.jersey}", "jerseyNum": player.jersey, "firstName": player.first_name, "lastName": player.last_name})
        players.append({"team_name": game.home_team_name, "teamid_label": 1, "team_goal_side": home_goalie_side, "first_name": player.first_name, "last_name": player.last_name, "jersey_number": int(player.jersey)})
    for player in game.away_players:
        game_request["requestDetails"]["rosters"]["teams"][1]["players"].append({"playerRefId": f"A{player.jersey}", "jerseyNum": player.jersey, "firstName": player.first_name, "lastName": player.last_name})
        players.append({"team_name": game.away_team_name, "teamid_label": 2, "team_goal_side": away_goalie_side, "first_name": player.first_name, "last_name": player.last_name, "jersey_number": int(player.jersey)})

    return game_request, players


def main() -> None:
    parser = argparse.ArgumentParser(description='Run detection on hockey broadcast videos.')
    parser.add_argument('-g', '--gameid', required=True, type=int, help='LHDRS Game Id')
    parser.add_argument('--goalieonleft', required=True, type=str, choices=["home", "away"], help='Which teams goalie is on left at faceoff')
    parser.add_argument('--faceofftime', required=True, type=int, help='Video timestamp of the 1st faceoff. In seconds')
    parser.add_argument('--whistletime', required=True, type=int, help='Video timestamp of the last whistle. In seconds')
    parser.add_argument('--outrequest', required=True, type=str, help='JSON output filename')
    parser.add_argument('--outplayers', required=True, type=str, help='JSON player details output')
    parser.add_argument("-l", "--log", help="log level (default: info)", choices=["debug", "info", "warning", "error", "critical"], default="info")
    args = parser.parse_args()

    logdatefmt = '%Y%m%dT%H:%M:%S'
    logformat = '%(asctime)s.%(msecs)03d [%(levelname)s] -%(name)s- -%(threadName)s- : %(message)s'
    logging.basicConfig(datefmt=logdatefmt, format=logformat, level=args.log.upper())

    game_id = args.gameid
    game = parse_boxscore(game_id)

    game_request, player_details = build_request(game, args.faceofftime, args.whistletime, args.goalieonleft)

    with open(args.outrequest, "w", encoding='utf8') as f:
        json.dump(game_request, f, indent=4, ensure_ascii=False)

    with open(args.outplayers, "w", encoding='utf8') as f:
        for player in player_details:
            data = json.dumps(player, ensure_ascii=False)
            print(data, file=f)


if __name__ == '__main__':
    main()
