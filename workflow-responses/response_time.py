import time
import datetime as dt
import matplotlib.pyplot as plt


def parse_duration(time_str):
    time_str = time_str.strip()
    if 'h' in time_str and 'm' in time_str and 's' in time_str:
        hours, minutes_seconds = time_str.split('h')
        hours = int(hours.strip())
        minutes, seconds = minutes_seconds.split('m')
        minutes = int(minutes.strip())
        seconds = int(seconds.strip().replace('s', ''))
        return hours * 3600 + minutes * 60 + seconds
    elif 'm' in time_str and 's' in time_str:
        minutes, seconds = time_str.split('m')
        minutes = int(minutes.strip())
        seconds = int(seconds.strip().replace('s', ''))
        return minutes * 60 + seconds
    elif 'm' in time_str:
        minutes = int(time_str.replace('m', '').strip())
        return minutes * 60
    elif 's' in time_str:
        seconds = int(time_str.replace('s', '').strip())
        return seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")


filename = 'log-data/ImportOpvlWaterCGOO2h-20260316.txt'
with open(filename, 'r') as file:
    lines = file.readlines()

# split line on tab
times = []
duration = []

for line in lines:
    parts = line.split('\t')
    if len(parts) >= 2:
        x, y = parts[1], parts[-1].strip('\n')
        # parse this format: '05/03/2026 13:02:46 GMT'
        x = x.replace(' GMT', '')
        x = dt.datetime.strptime(x, '%d/%m/%Y %H:%M:%S')
        times.append(x)
        y = y.replace('WorkflowFiles/Imports/wf_ImportOpvlWaterCGOO2h.xml completed in ', '')
        duration.append(parse_duration(y))

fig, ax = plt.subplots()
ax.plot(times, [d / 60 for d in duration], marker='o')
# rotate x-axis labels for better readability
plt.xticks(rotation=45)
ax.set_xlabel('Time')
ax.set_ylabel('Duration (minutes)')
ax.set_title('wf_ImportOpvlWaterCGOO2h response times')
ax.grid()
plt.tight_layout()
plt.savefig(f'figures/{int(time.time())}response_time.png')
