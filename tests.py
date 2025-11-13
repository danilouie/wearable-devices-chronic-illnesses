from load import get_chronic_data

def not_empty(self):
    chronic_data = get_chronic_data()
    with open(chronic_data, 'r') as f:
        lines = f.readlines()
        self.assertGreater(len(lines), 1, "CSV file does not contain data rows")

if __name__ == "__main__":
    not_empty()
