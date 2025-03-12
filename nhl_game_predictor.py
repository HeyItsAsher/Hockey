import pandas as pd
import numpy as np
import requests
# Nhl game predictor that uses a simple formula to predict the outcome of a game based on the team's performance in the season so far
#  just a simple formula with adjustable weights 
class NHLGamePredictor:
    def __init__(self):
        self.CurrentSeason = "20242025"
        # Prediction weights for different factors
        self.Weights = {
            'PointsPct': 0.45,   # Season performance (point percentage)
            'GoalDiff': 0.30,    # Scoring effectiveness
            'PowerPlay': 0.125,  # Power play impact
            'PenaltyKill': 0.125 # Penalty kill impact
        }
        
    def FetchTeamStats(self):
        baseUrl = "https://api.nhle.com/stats/rest/en/team/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D%5D&start=0&limit=50&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId="
        fullUrl = f"{baseUrl}{self.CurrentSeason}"
        
        try:
            response = requests.get(fullUrl)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('data'): 
                raise ValueError(f"No data available for season {self.CurrentSeason}")
                
            return pd.DataFrame([self.ProcessTeamStats(team) for team in data['data']])

        except requests.RequestException as e:
            print(f"Error fetching data from NHL API: {e}")
            return pd.DataFrame()
        except ValueError as e:
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def ProcessTeamStats(self, team):
        # puts stats into a dictionary
        shotsAgainstPerGame = float(team.get('shotsAgainstPerGame', 0))
        goalsAgainstPerGame = float(team.get('goalsAgainstPerGame', 0))
        
        if shotsAgainstPerGame > 0:
            savePercentage = (shotsAgainstPerGame - goalsAgainstPerGame) / shotsAgainstPerGame
        else:
            savePercentage = 0
        
        stats = {
            'TeamId': team['teamId'],
            'TeamName': team['teamFullName'],
            'GamesPlayed': team['gamesPlayed'],
            'Wins': team['wins'],
            'Losses': team['losses'],
            'OtLosses': team['otLosses'],
            'Points': team['points'],
            'GoalsFor': team['goalsFor'],
            'GoalsAgainst': team['goalsAgainst'],
            'PowerPlayPct': float(team.get('powerPlayPct', 0)) * 100,
            'PenaltyKillPct': float(team.get('penaltyKillPct', 0)) * 100,
            'SavePct': savePercentage
        }
        
        games = stats['GamesPlayed']
        stats.update({
            'WinPct': (stats['Wins'] / games) * 100,
            'PointsPct': (stats['Points'] / (games * 2)) * 100,
            'GoalsForPerGame': stats['GoalsFor'] / games,
            'GoalsAgainstPerGame': stats['GoalsAgainst'] / games,
            'GoalDifferentialPerGame': (stats['GoalsFor'] - stats['GoalsAgainst']) / games
        })
        
        return stats

    def FindTeamByName(self, teamStats, teamName):
        teamName = teamName.lower()
        matches = teamStats[teamStats['TeamName'].str.lower().str.contains(teamName)]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def GetTeamScore(self, team):
        pointFactor = team['PointsPct'] / 100  # Convert percentage to decimal
        # Scale goal differential to -1 to 1 range for proper probability conversion
        # divided by 4 to limit the goal differential to a more reasonable range
        goalDiff = np.clip(team['GoalDifferentialPerGame'] / 4.0, -1, 1)
        ppScore = min(team['PowerPlayPct'] / 20.0, 1)
        pkScore = min(team['PenaltyKillPct'] / 80.0, 1)
        
        return (self.Weights['PointsPct'] * pointFactor +
               self.Weights['GoalDiff'] * ((goalDiff + 1) / 2) +
               self.Weights['PowerPlay'] * ppScore +
               self.Weights['PenaltyKill'] * pkScore +
               np.random.normal(0, 0.02))

    def CalculateWinProbability(self, homeStats, awayStats):
        homeScore = self.GetTeamScore(homeStats)
        awayScore = self.GetTeamScore(awayStats)
        # sigmoid function to calculate the win probability
        prob = 1 / (1 + np.exp(-5 * (homeScore - awayScore)))
        return np.clip(prob, 0.15, 0.85)

    def PredictGame(self, homeTeam, awayTeam):
        prob = self.CalculateWinProbability(homeTeam, awayTeam)
        return {
            'HomeTeamWinProbability': prob,
            'AwayTeamWinProbability': 1 - prob
        }

    def PrintTeamStats(self, team):
        print(f"\n{team['TeamName']}:")
        print(f"Record: {team['Wins']}-{team['Losses']}-{team['OtLosses']}")
        print(f"Points: {team['Points']} ({team['Points']/(team['GamesPlayed']*2):.1%} of possible)")
        print(f"Goals For/Game: {team['GoalsForPerGame']:.2f}")
        print(f"Goals Against/Game: {team['GoalsAgainstPerGame']:.2f}")
        print(f"Power Play: {team['PowerPlayPct']:.1f}%")
        print(f"Penalty Kill: {team['PenaltyKillPct']:.1f}%")
        print(f"Save Percentage: {team['SavePct']:.3f}")

def main():
    predictor = NHLGamePredictor()
    
    print("\nEnter game details:")
    date = input("Enter date (YYYY-MM-DD): ")
    homeTeam = input("Enter home team name: ")
    awayTeam = input("Enter away team name: ")
    
    teamStats = predictor.FetchTeamStats()
    home = predictor.FindTeamByName(teamStats, homeTeam)
    away = predictor.FindTeamByName(teamStats, awayTeam)
    
    if home is None:
        print(f"Could not find team matching '{homeTeam}'")
        return
    if away is None:
        print(f"Could not find team matching '{awayTeam}'")
        return
    
    prediction = predictor.PredictGame(home, away)
    
    if prediction:
        print(f"\nPrediction for {home['TeamName']} vs {away['TeamName']} on {date}:")
        print(f"{home['TeamName']} win probability: {prediction['HomeTeamWinProbability']:.1%}")
        print(f"{away['TeamName']} win probability: {prediction['AwayTeamWinProbability']:.1%}")
    
        print("\nTeam Statistics:")
        predictor.PrintTeamStats(home)
        predictor.PrintTeamStats(away)

if __name__ == "__main__":
    main() 