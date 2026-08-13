def chron_train_val_test(df, train_pct = 0.8, val_pct = 0.1):
    df = df.sort_values(['user_id','timestamp']).copy()
    group_size = df.groupby('user_id')['user_id'].transform('size')
    row_index = df.groupby('user_id').cumcount() #gives a count like 1, 2, 3, etc 

    pct_position = row_index / group_size 

    train_df = df[pct_position < train_pct]
    val_df = df[(pct_position >= train_pct) & (pct_position < train_pct+ val_pct)]
    test_df = df[pct_position >= train_pct + val_pct]
    
    return train_df, val_df, test_df