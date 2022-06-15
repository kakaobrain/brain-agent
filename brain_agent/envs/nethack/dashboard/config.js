// Copyright (c) Facebook, Inc. and its affiliates.
module.exports = {
  serverPort: 3000,
  data: {
    defaultPath: '../../nle_data/',
    defaultRunsToRead: 100,
    stats: 'stats.csv',
  },
  // Currently we assume there is no header in the stats.csv file.
  // Would be better to read the header directly from the file.
  statsHeaders: [
    'episode',
    'end_status',
    'score',
    'time',
    'steps',
    'hp',
    'exp',
    'exp_lev',
    'gold',
    'hunger',
    'ttyrec',
  ],
};
