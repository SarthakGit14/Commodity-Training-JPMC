from datetime import datetime

def price_storage_contract(
    injection_dates, 
    withdrawal_dates, 
    purchase_prices, 
    sale_prices, 
    volume, 
    inj_wth_rate_cost, 
    storage_cost_per_month, 
    max_volume
):

    inj_dates = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in injection_dates]
    wth_dates = [datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d for d in withdrawal_dates]
    
    if len(inj_dates) != len(purchase_prices):
        raise ValueError("Number of injection dates must match number of purchase prices.")
    if len(wth_dates) != len(sale_prices):
        raise ValueError("Number of withdrawal dates must match number of sale prices.")
        
    total_injected = volume * len(inj_dates)
    total_withdrawn = volume * len(wth_dates)
    
    if total_injected > max_volume:
        raise ValueError("Total injected volume exceeds maximum storage capacity.")
    if total_withdrawn > total_injected:
        raise ValueError("Cannot withdraw more gas than what was injected.")

    total_cash_flow = 0.0
    
    # Calculate Injection (Purchasing & moving gas into storage)
    for price in purchase_prices:
        gas_purchase_cost = volume * price
        injection_cost = volume * inj_wth_rate_cost
        total_cash_flow -= (gas_purchase_cost + injection_cost)
        
    # Calculate Withdrawals (Selling & moving gas out of storage)
    for price in sale_prices:
        gas_sale_revenue = volume * price
        withdrawal_cost = volume * inj_wth_rate_cost
        total_cash_flow += gas_sale_revenue
        total_cash_flow -= withdrawal_cost
        
    # Calculate Time-Based Storage Costs
    # Lease starts on the first injection day and ends on the final withdrawal day
    first_injection = min(inj_dates)
    last_withdrawal = max(wth_dates)
    
    duration_days = (last_withdrawal - first_injection).days
    duration_months = duration_days / 30.4375  # Standardize to average days per month
    
    total_storage_cost = storage_cost_per_month * duration_months
    total_cash_flow -= total_storage_cost
    
    return total_cash_flow


# ==========================================
# TEST SCRIPT: Pricing a Sample Contract
# ==========================================

# Scenario: We inject gas twice over the summer (when prices are lower) 
# and withdraw it twice over the winter (when prices are higher).
in_dates = ['2023-06-01', '2023-07-01']
out_dates = ['2024-01-01', '2024-02-01']

# Prices mapped directly to the dates above (can be pulled from our prior `get_price` model)
in_prices = [10.0, 10.2]  # $10.00 and $10.20 per MMBtu
out_prices = [12.5, 12.8] # $12.50 and $12.80 per MMBtu

# Parameters for the physical storage limits and rates
volume_per_action = 10000   # 10,000 MMBtu per date
inj_wth_rate_cost = 0.05    # $0.05 fee per MMBtu to inject or withdraw
storage_cost = 5000         # $5,000 monthly rental fee
max_capacity = 50000        # Facility max capacity

# Execute the pricing model
contract_value = price_storage_contract(
    injection_dates=in_dates,
    withdrawal_dates=out_dates,
    purchase_prices=in_prices,
    sale_prices=out_prices,
    volume=volume_per_action,
    inj_wth_rate_cost=inj_wth_rate_cost,
    storage_cost_per_month=storage_cost,
    max_volume=max_capacity
)

print(f"Contract Net Value: ${contract_value:,.2f}")