"""Synthetic sealed-unit auction and margin-clearing warehouse.

No real payment, bank, encryption, tax or credit functionality is used.
All money is integer cents and all identities/auctions are fictional.
"""

from .db import Database
from .margin import MarginLedger
from .auction import AuctionService
from .clearing import ClearingService

__all__ = ["Database", "MarginLedger", "AuctionService", "ClearingService"]
